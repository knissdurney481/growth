from growth_twin.compilation.traverse import curriculum_lambda_rank, physics_gate


def test_lambda_rank_phases() -> None:
    assert curriculum_lambda_rank(10, 20, 50, 0.5) == 0.0
    assert abs(curriculum_lambda_rank(35, 20, 50, 0.5) - 0.25) < 1e-9
    assert abs(curriculum_lambda_rank(50, 20, 50, 0.5) - 0.5) < 1e-9
    assert abs(curriculum_lambda_rank(80, 20, 50, 0.5) - 0.5) < 1e-9


def test_physics_gate_activation() -> None:
    assert physics_gate(10, 20, 50) == (True, False, False)
    assert physics_gate(35, 20, 50) == (True, False, False)
    assert physics_gate(60, 20, 50) == (True, True, True)
