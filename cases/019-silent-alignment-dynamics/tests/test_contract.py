"""Small independent checks, kept separate from the 17-trajectory pilot."""

import tempfile
import unittest
from pathlib import Path
import numpy as np

from silent_alignment import math as M
from silent_alignment.analysis import crossing, reduce_trajectory
from silent_alignment.engine import make_inputs
from silent_alignment.io import Budget, assert_bundle, bundle, check_files, fresh_output, read_json, seal_files
from silent_alignment.solvers import dop853, rk4, scalar_reference

ROOT = Path(__file__).resolve().parents[1]
CONFIG = read_json(ROOT / "configs/p0.json")
TOL = CONFIG["tolerances"]


class Contracts(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(np.random.SeedSequence(CONFIG["streams"]["tests"]))

    def test_finite_difference_gradient(self):
        C, U, V = (self.rng.normal(size=s) for s in ((3, 4), (3, 2), (4, 2)))
        du, dv = M.physical_rhs(C, U, V)
        for x, grad in ((U, -du), (V, -dv)):
            for index in np.ndindex(x.shape):
                old = x[index]
                x[index] = old + 1e-6
                hi = M.loss(C, U, V)
                x[index] = old - 1e-6
                lo = M.loss(C, U, V)
                x[index] = old
                self.assertAlmostEqual((hi-lo)/2e-6, grad[index], places=7)

    def test_whitened_dataset_loss_and_gradient(self):
        d = 4
        C, U, V = (self.rng.normal(size=s) for s in ((3, d), (3, 2), (d, 2)))
        X = np.sqrt(d)*np.eye(d)
        Y = C @ X
        E = U @ V.T @ X - Y
        self.assertAlmostEqual(.5*np.sum(E**2)/d, M.loss(C, U, V), places=12)
        du, dv = M.physical_rhs(C, U, V)
        np.testing.assert_allclose(E @ X.T @ V/d, -du, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(X @ E.T @ U/d, -dv, rtol=1e-13, atol=1e-13)

    def test_rectangular_shape_and_no_time_normalization(self):
        C, V = np.ones((3, 4)), np.ones((4, 2))
        U = np.zeros((3, 2))
        M.validate(C, U, V)
        du, dv = M.physical_rhs(C, U, V)
        np.testing.assert_array_equal(du, np.full((3, 2), 4.0))
        np.testing.assert_array_equal(dv, np.zeros((4, 2)))
        with self.assertRaisesRegex(ValueError, "SHAPE"):
            M.validate(C, U, np.ones((3, 2)))

    def test_fp64_finite_contract(self):
        with self.assertRaisesRegex(ValueError, "FP64"):
            M.validate(np.eye(2, dtype=np.float32), np.eye(2), np.eye(2))
        with self.assertRaisesRegex(ValueError, "FP64"):
            M.validate(np.full((2, 2), np.nan), np.eye(2), np.eye(2))

    def test_haar_qr_reproducibility_and_orthogonality(self):
        R = M.stiefel(8, 4, [61023019, 99, 1])
        np.testing.assert_allclose(R.T @ R, np.eye(4), atol=1e-14)
        np.testing.assert_array_equal(R, M.stiefel(8, 4, [61023019, 99, 1]))
        self.assertFalse(np.array_equal(R, M.stiefel(8, 4, [61023019, 99, 2])))

    def test_stiefel_invalid_shape(self):
        with self.assertRaisesRegex(ValueError, "STIEFEL"):
            M.stiefel(4, 5, [1])

    def test_rho_amplitude_and_initial_imbalance(self):
        for rho in (1e-2, 1e-4, 1e-6):
            R = M.stiefel(8, 4, [12])
            V = np.sqrt(2*rho)*R
            # Test Gram at its own scale: accumulation/QR roundoff is relative
            # to 2*rho, including entries whose exact value is zero.
            np.testing.assert_allclose((V.T @ V)/(2*rho), np.eye(4), rtol=0,
                                       atol=64*np.finfo(np.float64).eps*R.shape[0])
            self.assertAlmostEqual(np.linalg.norm(V)**2, 2*rho*4, places=15)

    def test_random_aligned_initial_norm_loss_imbalance(self):
        C = np.diag(np.arange(8, 0, -1).astype(float))
        random, aligned = M.stiefel(8, 4, [13]), np.eye(8)[:, :4]
        for R in (random, aligned):
            U, V = np.zeros((8, 4)), np.sqrt(2e-4)*R
            self.assertAlmostEqual(M.loss(C, U, V), 102.0, places=12)
            self.assertAlmostEqual(np.linalg.norm(V)**2, .0008, places=15)

    def test_scaled_original_rhs_equivalence(self):
        rho = 1e-4
        C, A, B = (self.rng.normal(size=s) for s in ((3, 4), (3, 2), (4, 2)))
        du, dv = M.physical_rhs(C, np.sqrt(2*rho)*A, np.sqrt(2*rho)*B)
        da, db = M.scaled_rhs(C, A, B, rho)
        np.testing.assert_allclose(du, np.sqrt(2*rho)*da, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(dv, np.sqrt(2*rho)*db, rtol=1e-13, atol=1e-14)

    def test_projector_rotation_and_sign_invariance(self):
        R = M.stiefel(8, 4, [2])
        target = np.eye(8)[:, :4]
        Q = M.stiefel(4, 4, [4])
        before = M.subspace(R, target, TOL)[0]
        after = M.subspace(R@Q, target, TOL)[0]
        self.assertAlmostEqual(before, after, places=14)
        self.assertAlmostEqual(before, M.subspace(-R, target, TOL)[0], places=14)

    def test_zero_left_direction_na_and_right_valid(self):
        C = np.diag([1., .8])
        R = np.ones((2, 1))/np.sqrt(2)
        row = M.metrics(C, np.eye(2), np.array([1., .8]), np.eye(2), np.zeros((2, 1)), R, 1e-6, TOL)
        self.assertTrue(np.isnan(row["left_distance"]))
        self.assertEqual(row["left_rank"], 0)
        self.assertAlmostEqual(row["right_distance"], .5)
        self.assertEqual(row["z_frobenius"], 0)

    def test_rank_deficiency_has_no_qr_completion(self):
        F = np.c_[np.arange(1., 5.), np.arange(1., 5.)]
        d, angles, rank, condition = M.subspace(F, np.eye(4)[:, :2], TOL)
        self.assertEqual(rank, 1)
        self.assertTrue(np.isnan(d) and np.isnan(angles).all() and np.isnan(condition))

    def test_low_amplitude_scaled_direction(self):
        R = M.stiefel(8, 4, [8])
        before = M.subspace(R, np.eye(8)[:, :4], TOL)[0]
        # Scaled factors retain meaningful direction when physical amplitude is tiny.
        Uphysical = 1e-20*R
        self.assertTrue(np.isnan(M.subspace(Uphysical, np.eye(8)[:, :4], TOL)[0]))
        self.assertAlmostEqual(M.subspace(Uphysical/1e-20, np.eye(8)[:, :4], TOL)[0], before)

    def test_boundary_degeneracy_is_flagged(self):
        sigma = np.array([1., .8, .8, .2])
        self.assertFalse(M.target_diagnostics(sigma, 2, TOL)["target_subspace_identifiable"])
        row = M.metrics(np.diag(sigma), np.eye(4), sigma, np.eye(4), np.eye(4)[:, :2], np.eye(4)[:, :2], 1e-4, TOL)
        self.assertTrue(np.isnan(row["right_distance"]))
        near = M.target_diagnostics(np.array([1., .8, .8-1e-12, .2]), 2, TOL)
        self.assertTrue(near["boundary_gap_warning"])

    def test_target_modes_not_sorted_singular_values(self):
        sigma = np.array([1., .5])
        A, B = np.array([[0.], [1.]]), np.array([[1.], [0.]])
        row = M.metrics(np.diag(sigma), np.eye(2), sigma, np.eye(2), A, B, .5, TOL)
        np.testing.assert_allclose(row["target_mode_signed"], [0., 0.])
        self.assertAlmostEqual(row["z_singular_sorted"][0], 1.)

    def test_rank_limited_loss_floor(self):
        C = np.diag([1., .9, .4, .3])
        U = np.eye(4)[:, :2]*np.sqrt([1., .9])
        V = U.copy()
        self.assertAlmostEqual(M.loss(C, U, V), .125)
        self.assertGreater(M.loss(C, U, V), 0.)

    def test_zero_target_stationary(self):
        C, R = np.zeros((3, 3)), np.eye(3)[:, :2]
        A, B, _ = dop853(C, R, .01, np.array([0., 1.]), CONFIG["solver"], Budget(10))
        np.testing.assert_array_equal(A, np.zeros_like(A))
        np.testing.assert_allclose(B[1], R, atol=1e-15)

    def test_short_original_reference_and_invariants(self):
        C, R = np.diag([1., .8]), np.ones((2, 1))/np.sqrt(2)
        t = np.linspace(0, 1, 11)
        A, B, _ = dop853(C, R, 1e-4, t, CONFIG["solver"], Budget(10))
        RA, RB, _ = rk4(C, R, 1e-4, t, .005, Budget(10))
        np.testing.assert_allclose(2e-4*A@B.transpose(0, 2, 1), 2e-4*RA@RB.transpose(0, 2, 1), rtol=1e-7, atol=1e-11)
        losses = [M.loss(C, np.sqrt(2e-4)*a, np.sqrt(2e-4)*b) for a,b in zip(A, B)]
        self.assertLessEqual(np.max(np.diff(losses)), 0.)
        for a,b in zip(A, B):
            np.testing.assert_allclose(2e-4*(b.T@b-a.T@a), [[2e-4]], rtol=1e-9)

    def test_scalar_relation_and_early_approximation(self):
        rho, sigma, u = 1e-4, .5, .03
        v = np.sqrt(u*u+2*rho)
        du, dv = (sigma-u*v)*v, (sigma-u*v)*u
        self.assertAlmostEqual(du*v+u*dv, M.scalar_rhs(0, u*v, sigma, rho), places=14)
        t = np.linspace(0, .1, 6)
        z, _ = scalar_reference(sigma, rho, t, Budget(10))
        np.testing.assert_allclose(z, rho*np.sinh(2*sigma*t), rtol=5e-5, atol=1e-12)

    def test_closed_form_gram_and_condition(self):
        C = np.diag([1., .6])
        R = np.array([[.6], [.8]])
        A, B, _ = dop853(C, R, 1e-4, np.array([0., .5]), CONFIG["solver"], Budget(10))
        W = np.sqrt(2e-4)*np.vstack([A[-1], B[-1]])
        exact, info = M.closed_gram(C, R, 1e-4, .5)
        np.testing.assert_allclose(W@W.T, exact, rtol=1e-9, atol=1e-13)
        self.assertTrue(info["finite"])
        self.assertLess(info["solve_condition"], 2.)

    def test_small_rho_limit_is_approximation_not_gate(self):
        C, R = np.diag([1., .6]), np.array([[.6], [.8]])
        errors = []
        for rho in (.01, 1e-6):
            A, B, _ = dop853(C, R, rho, np.array([0., 1.]), CONFIG["solver"], Budget(10))
            u, v = M.linearized_factors(C, R, rho, 1.)
            errors.append(np.linalg.norm(2*rho*A[-1]@B[-1].T-u@v.T)/np.linalg.norm(u@v.T))
        self.assertLess(errors[1], errors[0])
        self.assertGreater(errors[0], 1e-5)

    def test_crossings_and_unreached_are_explicit(self):
        t = np.array([0., 1., 2.])
        c = crossing(t, np.array([1., .8, .5]), .9)
        self.assertAlmostEqual(c["time"], .5)
        self.assertEqual(c["bracket"], [0., 1.])
        never = crossing(t, np.array([1., .99, .98]), .9)
        self.assertEqual(never["status"], "NOT_REACHED")
        self.assertIsNone(never["time"])
        self.assertEqual(crossing(t, np.zeros(3), 0., already=True)["time"], 0.)

    def test_17_keys_pairing_and_unique_aligned_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_inputs(CONFIG, Path(tmp))
            ts = plan["trajectories"]
            self.assertEqual(len(ts), 17)
            self.assertEqual([sum(x["block"] == b for x in ts) for b in ("A", "B", "C")], [6, 9, 2])
            self.assertEqual(sum(x["initialization"] == "aligned" and x["block"] == "B" for x in ts), 3)
            group = [x for x in ts if x["block"] == "B" and x["initialization"] == "random" and x["replicate"] == 0]
            self.assertEqual(len({x["R_sha256"] for x in group}), 1)
            self.assertEqual(len({x["target_sha256"] for x in group}), 1)

    def test_output_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                fresh_output(tmp)

    def test_source_and_config_tamper_rejected(self):
        for relative in ("src/a.py", "configs/p0.json"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                p = root/relative
                p.parent.mkdir(parents=True)
                p.write_text("before")
                frozen = bundle(root)
                p.write_text("after")
                with self.assertRaisesRegex(ValueError, "SOURCE_OR_CONFIG_CHANGED"):
                    assert_bundle(root, frozen)

    def test_raw_bytes_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"a").write_text("old")
            rows = seal_files(root)
            (root/"a").write_text("new")
            with self.assertRaisesRegex(ValueError, "BYTES_CHANGED"):
                check_files(root, rows)

    def test_fixed_step_misaligned_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "GRID"):
            rk4(np.eye(1), np.eye(1), .01, np.array([0., .015]), .01, Budget(10))


if __name__ == "__main__":
    unittest.main()
