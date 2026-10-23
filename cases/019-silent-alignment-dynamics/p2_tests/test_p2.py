"""P2-only contract and mathematical fixtures, separate from128 primary paths."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from silent_alignment.io import Budget,read_json,check_files,entry,write_json
from silent_alignment.math import stiefel,physical_rhs
from silent_alignment.p1.metrics import direction
from silent_alignment.p2.contract import grid,checkpoints,time_join,planned,source_bundle,WorkerBudget
from silent_alignment.p2.core import (log_filter,graph_setup,graph_metrics,rotations,transform,
                                     matching,integrate,reference,relative_error)
from silent_alignment.p2.diagnostics import persistence
from silent_alignment.p2.analysis import effect

ROOT=Path(__file__).resolve().parents[1]
CFG=read_json(ROOT/"configs/p2.json");TOL=CFG["tolerances"]


class P2Contracts(unittest.TestCase):
    def fixture(self):
        C=np.diag([1.,.8,.4,.2]);left=right=np.eye(4)
        R=stiefel(4,2,[61023022,99]);rho=1e-4
        u,v,_=integrate(C,np.zeros((4,2)),np.sqrt(2*rho)*R,rho,np.array([0.,.3]),CFG["solver"],Budget(10))
        return C,left,right,np.diag(C),u[-1],v[-1],rho

    def test_finite_rho_graph_matches_direct_filtered_projector(self):
        R=stiefel(8,4,[61023022,98]);s=np.linspace(1.,.1,8);rho=.01;t=np.array([0.,.25,1.,5.])
        M,_=graph_setup(np.eye(8),R);mean,worst=graph_metrics(s,rho,t,M,Budget(5))
        omega=np.sqrt(s*s+rho*rho)
        for i,tt in enumerate(t):
            f=np.cosh(omega*tt)+rho/omega*np.sinh(omega*tt)
            d=direction(f[:,None]*R,np.eye(8)[:,:4],TOL)
            self.assertAlmostEqual(mean[i],d["distance"],places=12);self.assertAlmostEqual(worst[i],d["worst"],places=12)

    def test_graph_matches_finite_rho_original_ODE(self):
        s=np.array([1.,.8,.4,.2]);R=stiefel(4,2,[61023022,97]);rho=.1;t=np.array([0.,.2,.5,1.])
        u,v,_=integrate(np.diag(s),np.zeros((4,2)),np.sqrt(2*rho)*R,rho,t,CFG["solver"],Budget(10))
        M,_=graph_setup(np.eye(4),R);mean,worst=graph_metrics(s,rho,t,M,Budget(5))
        for i in range(len(t)):
            d=direction(v[i],np.eye(4)[:,:2],TOL)
            self.assertAlmostEqual(mean[i],d["distance"],places=9);self.assertAlmostEqual(worst[i],d["worst"],places=9)

    def test_log_filter_no_cosh_overflow_and_zero_time(self):
        s=np.array([1.,.2,0.]);rho=.01;t=np.array([0.,1.,10000.])
        a=log_filter(s,rho,t)
        np.testing.assert_allclose(a[0],0,atol=1e-15,rtol=0);self.assertTrue(np.isfinite(a).all())
        np.testing.assert_allclose(a[1],np.log(np.cosh(np.hypot(s,rho))+rho/np.hypot(s,rho)*np.sinh(np.hypot(s,rho))),atol=1e-14)

    def test_graph_rank_and_condition_failure_no_pseudoinverse(self):
        R=np.eye(4)[:,2:]
        with self.assertRaisesRegex(ValueError,"RANK_OR_CONDITION"):graph_setup(np.eye(4),R)
        R=np.array([[1e-8,0],[0,1],[1,0],[0,0.]])
        with self.assertRaisesRegex(ValueError,"CONDITION"):graph_setup(np.eye(4),R,max_condition=1e6)

    def test_four_arm_gram_size_and_oracle_matching(self):
        C,l,r,s,u,v,rho=self.fixture();rot=rotations(61023022,2,4)
        for arm in ["CONTINUE","GAUGE","ALIGN","ROTATE"]:
            up,vp=transform(u,v,l,r,arm,rot)
            m=matching(C,l,r,s,u,v,up,vp,arm,rot,rho,TOL)
            self.assertTrue(m["pass"],m)
        up,vp=transform(u,v,l,r,"ALIGN",rot)
        self.assertLessEqual(np.linalg.norm(up@vp.T-C),np.linalg.norm(u@v.T-C)+1e-12)

    def test_actual_nonisotropic_drift_preserved_not_corrected(self):
        C,l,r,s,u,v,rho=self.fixture();v=v.copy();v[0,0]+=1e-7;imb=v.T@v-u.T@u
        for arm in ["ALIGN","ROTATE"]:
            up,vp=transform(u,v,l,r,arm,rotations(61023022,2,4))
            np.testing.assert_allclose(vp.T@vp-up.T@up,imb,atol=1e-14,rtol=1e-10)
        self.assertGreater(np.linalg.norm(imb-2*rho*np.eye(2)),1e-12)

    def test_gauge_equivariance_and_nonzero_initial_paths(self):
        C,l,r,s,u,v,rho=self.fixture();rot=rotations(61023022,2,4);up,vp=transform(u,v,l,r,"GAUGE",rot)
        a,b=physical_rhs(C,u,v);aa,bb=physical_rhs(C,up,vp)
        np.testing.assert_allclose(aa,a@rot["O"],atol=1e-14);np.testing.assert_allclose(bb,b@rot["O"],atol=1e-14)
        t=np.array([0.,.1,.2]);U,V,_=integrate(C,u,v,rho,t,CFG["solver"],Budget(10))
        GU,GV,_=integrate(C,up,vp,rho,t,CFG["solver"],Budget(10))
        np.testing.assert_array_equal(U[0],u)
        self.assertGreater(np.linalg.norm(U[0]),0)
        for i in range(len(t)):np.testing.assert_allclose(U[i]@V[i].T,GU[i]@GV[i].T,atol=1e-12,rtol=1e-9)

    def test_physical_RK4_arbitrary_checkpoint_matches_scaled(self):
        C,l,r,s,u,v,rho=self.fixture();t=np.array([0.,.1,.2])
        U,V,_=integrate(C,u,v,rho,t,CFG["solver"],Budget(10));RU,RV,_=reference(C,u,v,t,.005,Budget(10))
        for i in range(len(t)):np.testing.assert_allclose(U[i]@V[i].T,RU[i]@RV[i].T,atol=1e-11,rtol=1e-7)

    def test_rotation_streams_are_fixed_orthogonal_and_distinct(self):
        r=rotations(61023022);rr=rotations(61023022)
        for k,v in r.items():
            np.testing.assert_array_equal(v,rr[k]);np.testing.assert_allclose(v.T@v,np.eye(len(v)),atol=1e-14,rtol=0)
        self.assertFalse(np.array_equal(r["O_L"],r["O_R"]))

    def test_grid_checkpoint_and_absolute_time_exact_join(self):
        t=grid();self.assertEqual(len(t),1677);self.assertEqual(sum(t<=20),1002);self.assertEqual(t[-1],155)
        for star in [1,5]:
            for tt in checkpoints(star):self.assertIn(tt,t)
        a,b=time_join(np.array([0.,.02,.25,.5]),np.array([1.,1.02,1.24,1.5]),1)
        np.testing.assert_array_equal(a,[0,1,3]);np.testing.assert_array_equal(b,[0,1,3])

    def test_32_parents_128_keys_and_exact_R2_reference_selection(self):
        pp,rows=planned(CFG)
        self.assertEqual(len(pp),32);self.assertEqual(len(rows),128);self.assertEqual(len({r["key"] for r in rows}),128)
        self.assertEqual(rows,planned(CFG)[1])
        for p in pp:
            x=[r for r in rows if r["parent_key"]==p["parent_key"]]
            self.assertEqual(len(x),4);self.assertEqual(len({r["p1_key"] for r in x}),1)
        r2=next(r for r in rows if r["key"]=="g2_R2_t5_CONTINUE")
        self.assertEqual(r2["p1_key"],"g2_rho2_R2");self.assertEqual(r2["R_stream"],[61023020,1,2])
        self.assertEqual(len(CFG["reference_selection"]),6)

    def test_J_G_A_common_denominator_and_origin(self):
        r=effect(10,7,6,4,9)
        self.assertEqual(r["J"],3);self.assertEqual(r["G"],-1);self.assertEqual(r["A"],2)
        self.assertEqual(effect(10,7,10,7,9)["G"],0)
        self.assertAlmostEqual(r["A_normalized"],r["J_normalized"]+r["G_normalized"])

    def test_signed_cross_reexit_and_10_time_retention(self):
        p=persistence([0,2,4,6,8,20],[0,.2,.2,0,.2,.2])
        self.assertEqual(p["up_entry_count"],2);self.assertEqual(p["down_exit_count"],1)
        self.assertEqual(p["first_entry"]["time"],1);self.assertEqual(p["last_retained_entry"],7)
        self.assertEqual(p["first_sustained10"]["start"],7);self.assertEqual(p["first_sustained10"]["end"],17)
        self.assertIsNone(persistence([0,10,20],[0,-.2,-.3])["first_entry"]["time"])

    def test_persistence_already_end_only_short_and_never(self):
        self.assertEqual(persistence([0,10,20],[.2,.2,.2])["first_entry"]["status"],"ALREADY_ABOVE")
        x=persistence([0,10,20],[0,0,.1]);self.assertTrue(x["end_only_entry"]);self.assertTrue(x["insufficient_remaining_for10"])
        self.assertIsNone(x["first_sustained10"])
        x=persistence([0,10,20],[0,0,0]);self.assertIsNone(x["last_retained_entry"]);self.assertEqual(x["first_entry"]["status"],"NOT_REACHED")

    def test_raw_hash_and_unsafe_path_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);f=p/"raw";f.write_bytes(b"raw-before");row=entry(f,p);check_files(p,[row]);f.write_bytes(b"raw-after!")
            with self.assertRaisesRegex(ValueError,"CHANGED"):check_files(p,[row])
            with self.assertRaisesRegex(ValueError,"UNSAFE"):check_files(p,[dict(row,path="../raw")])

    def test_existing_output_refused_before_budget_or_ODE(self):
        from silent_alignment.p2.runner import run
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):run(ROOT,ROOT/"configs/p2.json",temp,Path(temp)/"budget",Path(temp))
            self.assertFalse((Path(temp)/"budget").exists())

    def test_budget_debt_and_unresolved_attempt_not_reset(self):
        import time
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"budget.json";b=WorkerBudget(p,900,189.4325839030007,time.monotonic())
            with self.assertRaisesRegex(ValueError,"UNRESOLVED"):WorkerBudget(p,900,189.4325839030007,time.monotonic())
            b.finish("PARTIAL");used=read_json(p)["p2_consumed_seconds"]
            b=WorkerBudget(p,900,189.4325839030007,time.monotonic());self.assertEqual(b.debt,used);b.finish("COMPLETE")
            self.assertGreater(read_json(p)["campaign_consumed_seconds"],189)

    def test_pinned_source_identity_and_manifest_hash_not_git_base(self):
        b=source_bundle(ROOT)
        self.assertEqual(b["p1_combined_bundle_sha256"],"69e91d0bb8b0cff7dd6ad6ca9738e2f6a5d79fd122ec5c06c2de6e71211017e9")
        self.assertNotEqual(b["bundle_sha256"],b["p1_combined_bundle_sha256"])
        self.assertEqual(len(b["read_only_p0_p1_files"]),25)


if __name__=="__main__":unittest.main()
