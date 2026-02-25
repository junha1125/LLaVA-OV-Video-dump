import os, math
import torch
import matplotlib.pyplot as plt
from transformers import TrainerCallback

# 아주 작은 tap 객체: 모듈 출력 텐서 보관 + retain_grad
class _FeatureTap:
    def __init__(self):
        self.tensor = None
        self.handle = None
    def attach(self, module):
        def _hook(_, __, out):
            if isinstance(out, tuple):  # 혹시 tuple이면 첫 원소 사용
                out = out[0]
            self.tensor = out
            if torch.is_tensor(out):
                out.retain_grad()  # <-- 핵심: backward 후 .grad에 σL/σf 저장
        self.handle = module.register_forward_hook(_hook)
        return self
    def grad(self):
        if self.tensor is None:
            return None
        else:
            return self.tensor.grad
    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

# 시각화 콜백: tap에서 grad 읽어서 저장만 담당
class GradVizCallback(TrainerCallback):
    def __init__(self, tap, every_n_steps=200, out_dir="grad_viz", reduce="l1"):
        self.tap = tap
        self.every_n_steps = every_n_steps
        self.out_dir = out_dir
        self.reduce = reduce
        os.makedirs(self.out_dir, exist_ok=True)

    def _reduce_tokens(self, g_tok):  # [T_img, D] -> [T_img]
        if self.reduce == "l2":
            return g_tok.pow(2).sum(dim=-1).sqrt()
        return g_tok.abs().sum(dim=-1)

    def _save_heatmaps(self, step, g):  # g: [B, T_img, D] or [T_img, D]
        if g is None:
            return
        g = g.detach().float().cpu()
        if g.dim() == 2:
            g = g.unsqueeze(0)  # -> [1, T_img, D]
        B, T, D = g.shape
        for b in range(B):
            s = self._reduce_tokens(g[b])   # [T_img]
            T_img = s.numel()
            grid = int(math.sqrt(T_img))
            if grid * grid != T_img:
                # 정사각형이 아니면 건너뜀 (필요 시 (H,W) 복원 로직 추가)
                continue
            heat = s.reshape(grid, grid)
            path = os.path.join(self.out_dir, f"step{step:06d}_b{b}.png")
            plt.figure()
            plt.imshow(heat.numpy(), interpolation="nearest")
            plt.title(f"|dL/df| per patch (step {step}, b{b})")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()

    def on_step_end(self, args, state, control, **kwargs):
        # 저장 간격 + rank 0만
        if (state.global_step % self.every_n_steps) != 0:
            return
        if args.process_index != 0:
            return
        self._save_heatmaps(state.global_step, self.tap.grad())

def unwraped_and_tapping(trainer, training_args):
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    # mm_projector 모듈 찾기~
    def _find_mm_projector(m):
        base = m.get_model() if hasattr(m, "get_model") and callable(m.get_model) else m
        if hasattr(base, "mm_projector"):
            return getattr(base, "mm_projector")
        for name, module in base.named_modules():
            if "mm_projector" in name:
                return module
        raise RuntimeError("mm_projector 모듈을 찾을 수 없습니다.")

    projector = _find_mm_projector(unwrapped)

    # tap 만들고 projector 출력에 hook 부착
    tap = _FeatureTap().attach(projector)

    # 시각화 콜백 등록 (저장은 콜백이 담당)
    viz_cb = GradVizCallback(
        tap=tap,
        every_n_steps=11,
        out_dir=os.path.join(training_args.output_dir, "grad_viz"),
        reduce="l1",  # 또는 "l2"
    )
    trainer.add_callback(viz_cb)
    return trainer

"""
    # Accelerate/DeepSpeed 래핑 해제된 "실모델"을 얻는다.
    from llava.train.hook import GradVizCallback, _FeatureTap
    # ---------- 여기부터 추가: hook 설치 ----------
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    # mm_projector 모듈 찾기
    def _find_mm_projector(m):
        base = m.get_model() if hasattr(m, "get_model") and callable(m.get_model) else m
        if hasattr(base, "mm_projector"):
            return getattr(base, "mm_projector")
        for name, module in base.named_modules():
            if "mm_projector" in name:
                return module
        raise RuntimeError("mm_projector 모듈을 찾을 수 없습니다.")

    projector = _find_mm_projector(unwrapped)

    # tap 만들고 projector 출력에 hook 부착
    tap = _FeatureTap().attach(projector)

    # 시각화 콜백 등록 (저장은 콜백이 담당)
    viz_cb = GradVizCallback(
        tap=tap,
        every_n_steps=10,
        out_dir=os.path.join(training_args.output_dir, "grad_viz"),
        reduce="l1",  # 또는 "l2"
    )
    trainer.add_callback(viz_cb)
    # ---------- 추가 끝 ----------
"""