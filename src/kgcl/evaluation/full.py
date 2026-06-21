from __future__ import annotations

from kgcl.evaluation.exact_match import EvaluationSpec, run_exact_match_evaluation


def run(args) -> int:
    return run_exact_match_evaluation(
        args,
        EvaluationSpec(
            beam_size=args.full_beam_size,
            step_beam_size=args.step_beam_size,
            top_k_values=(1, 3, 5, 10),
            include_max_frag=False,
            track_stereo=True,
            output_kind='full',
        ),
    )
