"""P1-only tests: no primary P1 trajectories or P0 pilot reruns."""

import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np

from silent_alignment.io import Budget,read_json,write_json
from silent_alignment.solvers import dop853,rk4
from silent_alignment.p1.contract import (ControlBudget,assert_source,grid,make_inputs,make_targets,planned,source_bundle)
from silent_alignment.p1.metrics import direction,measure,compare
from silent_alignment.p1.analysis import descriptives,event,grouping,summarize_raw
from silent_alignment.p1.runner import run,scalar_vector,scalar_comparison

ROOT=Path(__file__).resolve().parents[1]
CFG=read_json(ROOT/"configs/p1.json");TOL=CFG["tolerances"]


class P1Contracts(unittest.TestCase):
    def setUp(self):
        with np.load(ROOT/CFG["parent_target"],allow_pickle=False) as a:self.C0=a["C"]

    def test_boundary_pair_only_and_exact_baseline(self):
        targets,l,r,cost=make_targets(self.C0,CFG["gaps"],TOL)
        self.assertEqual(cost["C0_svd_calls"],1)
        self.assertEqual(targets[0]["C"].tobytes(),self.C0.tobytes())
        mask=np.ones(64,dtype=bool);mask[[31,32]]=False
        for t,g in zip(targets,CFG["gaps"]):
            np.testing.assert_allclose(t["sigma"][mask],targets[0]["sigma"][mask],atol=1e-12,rtol=0)
            np.testing.assert_allclose(t["sigma"][[31,32]],[.6+g/2,.6-g/2],atol=1e-12,rtol=0)
            self.assertLess(t["receipt"]["fixed_basis_representation_frobenius_error"],1e-12)

    def test_basis_overlap_and_target_specific_loss_floor(self):
        targets,l,r,_=make_targets(self.C0,CFG["gaps"],TOL)
        floors=[]
        for t in targets:
            self.assertLess(t["receipt"]["basis_overlap_max_error"],TOL["basis_overlap_absolute"])
            floors.append(t["receipt"]["loss_min"])
            self.assertAlmostEqual(t["receipt"]["loss_min"],.5*np.sum(t["sigma"][32:]**2),places=14)
        self.assertEqual(len(set(floors)),3)

    def test_long_grid_and_checkpoint_membership(self):
        t=grid(CFG)
        self.assertEqual(len(t),1702)
        self.assertEqual(sum(t<=20),1002)
        self.assertEqual(sum(t>20),700)
        self.assertEqual(t[-1],160)
        self.assertEqual(len(np.unique(t)),1702)
        for ck in CFG["time"]["checkpoints"]:self.assertIn(ck,t)

    def test_81_unique_keys_and_order_seed(self):
        rows=planned(CFG)
        self.assertEqual(len(rows),81)
        self.assertEqual(len({r["key"] for r in rows}),81)
        self.assertEqual(rows,planned(CFG))
        self.assertEqual([r["order_index"] for r in rows],list(range(81)))
        changed=dict(CFG,order_seed=61023022)
        self.assertNotEqual([r["key"] for r in rows],[r["key"] for r in planned(changed)])

    def test_exact_reference_selection(self):
        rows=planned(CFG)
        got={(r["initialization"],r["g"],r["rho"],r["R_index"]) for r in rows if r["matrix_reference"]}
        self.assertEqual(got,{tuple(x) for x in CFG["reference_selection"]})
        self.assertEqual(len(got),12)
        self.assertEqual(sum(r["scalar_reference"] for r in rows),9)

    def test_pairing_eight_R_and_unique_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=make_inputs(ROOT,CFG,Path(tmp))
            rows=manifest["planned_trajectories"]
            for rep in range(8):
                selected=[r for r in rows if r["R_index"]==rep]
                self.assertEqual(len(selected),9)
                self.assertEqual(len({r["R_asset_sha256"] for r in selected}),1)
                self.assertTrue(all(r["R_stream"]==[61023020,1,rep] for r in selected))
            self.assertEqual(len({r["R_asset_sha256"] for r in rows if r["initialization"]=="random"}),8)
            self.assertEqual(sum(r["initialization"]=="aligned" for r in rows),9)

    def test_random_aligned_equal_initial_norm_imbalance_and_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=make_inputs(ROOT,CFG,Path(tmp))["planned_trajectories"]
            for g in CFG["gaps"]:
                for rho in CFG["rhos"]:
                    selected=[r for r in rows if r["g"]==g and r["rho"]==rho]
                    self.assertEqual(len(selected),9)
                    norms=[r["initial_V_norm"] for r in selected]
                    self.assertLess(max(norms)-min(norms),1e-13)
                    self.assertEqual(len({r["loss_0"] for r in selected}),1)
                    for r in selected:np.testing.assert_allclose(np.asarray(r["initial_gram_v"])/(2*rho),np.eye(32),atol=1e-12,rtol=0)

    def test_mean_and_worst_are_distinct(self):
        target=np.eye(64)[:,:32];F=target.copy();F[:,31]=np.eye(64)[:,32]
        d=direction(F,target,TOL)
        self.assertAlmostEqual(d["distance"],1/32)
        self.assertAlmostEqual(d["worst"],1)
        self.assertAlmostEqual(d["theta_max"],np.pi/2)

    def test_rank_deficiency_and_zero_left_na(self):
        target=np.eye(64)[:,:32]
        d=direction(np.zeros((64,32)),target,TOL)
        self.assertEqual(d["rank"],0);self.assertTrue(np.isnan(d["distance"]))
        self.assertTrue(np.isnan(d["cosines"]).all())
        F=target.copy();F[:,31]=F[:,30]
        self.assertEqual(direction(F,target,TOL)["rank"],31)

    def test_aligned_roundoff_does_not_create_a_learning_time(self):
        target=np.linalg.qr(np.random.default_rng(6102302099).normal(size=(64,32)))[0]
        d=direction(target,target,TOL)
        self.assertLess(abs(d["worst"]),TOL["aligned_worst_absolute"])
        self.assertLess(d["distance"],TOL["aligned_distance"])
        result=event(np.array([0.,.02]),np.array([d["worst"],0.]),d["worst"]*.5,already=True)
        self.assertEqual(result["status"],"ALREADY_ALIGNED");self.assertEqual(result["time"],0)

    def test_censor_and_execution_failure_separate(self):
        never=event(np.array([0.,20.,160.]),np.array([1.,.9,.8]),.5)
        self.assertEqual(never["status"],"NOT_REACHED");self.assertIsNone(never["time"])
        self.assertEqual(never["observed_end"],160)
        rows=[dict(r,execution_status="COMPLETED",S=None,events={"T_worst_half":never}) for r in planned(CFG)]
        groups=grouping(rows,planned(CFG),CFG)
        random=next(g for g in groups if g["initialization"]=="random")
        self.assertEqual(random["statistics"]["T_worst_half"]["not_reached"],8)
        self.assertEqual(random["statistics"]["T_worst_half"]["execution_failed"],0)
        self.assertIsNone(random["statistics"]["T_worst_half"]["mean"])

    def test_late_bracket_width_not_exact_event(self):
        x=event(np.array([20.,20.2,20.4]),np.array([1.,.8,.6]),.9)
        self.assertAlmostEqual(x["time"],20.1)
        self.assertAlmostEqual(x["bracket_width"],.2)

    def test_signed_mode_not_absolute(self):
        e=event(np.array([0.,1.,2.]),-np.array([0.,-.2,-.5]),-.1)
        self.assertEqual(e["status"],"NOT_REACHED")

    def test_descriptive_censor_is_not_filled(self):
        x=descriptives([1.,3.,None,None],8)
        self.assertEqual(x["reached"],2);self.assertEqual(x["total"],8)
        self.assertEqual(x["mean"],2)
        self.assertAlmostEqual(x["sample_sd"],np.sqrt(2))

    def test_cumulative_budget_keeps_prior_debt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"budget.json"
            b=ControlBudget(p,1800,"fixture1");b.finish("PARTIAL")
            previous=read_json(p)["consumed_seconds"]
            b=ControlBudget(p,1800,"fixture2")
            self.assertEqual(b.debt,previous)
            b.finish("COMPLETE")
            self.assertEqual(len(read_json(p)["attempts"]),2)

    def test_unresolved_budget_attempt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"budget.json";ControlBudget(p,1800,"fixture")
            with self.assertRaisesRegex(ValueError,"UNRESOLVED"):ControlBudget(p,1800,"fixture2")

    def test_source_config_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for d in ["src","tests","configs","specs","p1_tests"]:shutil.copytree(ROOT/d,root/d)
            for name in ["P1_AMENDMENT_01.md","P1_SOURCE_MAP.md"]:shutil.copyfile(ROOT/name,root/name)
            p=root/"results/p0_20261023-095100";p.mkdir(parents=True)
            shutil.copyfile(ROOT/"results/p0_20261023-095100/execution_freeze.json",p/"execution_freeze.json")
            original=source_bundle(root)
            q=root/"configs/p1.json";q.write_text(q.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"P1_SOURCE"):assert_source(root,original)

    def test_existing_output_rejected_before_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):run(ROOT,ROOT/"configs/p1.json",tmp,Path(tmp)/"budget.json")

    def test_scalar_vector_is_independent_factor_check(self):
        sigma=np.linspace(1.,.2,32);C=np.diag(sigma);R=np.eye(32);times=np.array([0.,.1,.2]);rho=1e-4
        A,B,_=dop853(C,R,rho,times,CFG["solver"],Budget(20))
        z,_=scalar_vector(sigma,rho,times,Budget(20))
        arrays,result=scalar_comparison(C,np.eye(32),np.eye(32),sigma,rho,A,B,z,TOL,Budget(20))
        self.assertTrue(result["pass"])
        np.testing.assert_array_equal(z[0],np.zeros(32))

    def test_compact_loss_spectrum_and_mode_recombination(self):
        sigma=np.linspace(1.,.2,64);C=np.diag(sigma)
        A=np.eye(64)[:,:32]*.3;B=np.eye(64)[:,:32]*1.2
        r=measure(C,np.eye(64),np.eye(64),sigma,A,B,1e-4,TOL)
        loss=.5*np.sum(C*C)+.5*np.sum(r["z_singular_sorted"]**2)-r["target_mode_signed"]@sigma
        self.assertAlmostEqual(loss,r["loss"],places=12)
        self.assertAlmostEqual(r["z_frobenius"]**2,r["gram_cross_trace"],places=15)


if __name__=="__main__":unittest.main()
