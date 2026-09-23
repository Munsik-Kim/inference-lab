"""Portable stdlib inventory tests: corruption, unsafe paths and hash identities."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.verify_publication import (PublicationError,build_manifest,canonical,main,sha,verify_manifest)


class PublicationInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        (self.root/"README.md").write_text("public artifact\n")
        (self.root/"data.bin").write_bytes(b"\x00\x01\x02")

    def write_manifest(self,manifest=None):
        manifest=build_manifest(self.root) if manifest is None else manifest
        (self.root/"publication-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        return manifest

    def test_exact_inventory_no_self_hash_and_relocatable(self):
        manifest=self.write_manifest()
        self.assertEqual(manifest["file_count"],2)
        self.assertEqual({row["path"] for row in manifest["files"]},{"README.md","data.bin"})
        self.assertNotIn("manifest_sha256",manifest)
        self.assertNotIn(str(self.root),json.dumps(manifest))
        result=verify_manifest(self.root)
        self.assertEqual(result["status"],"PASS")
        self.assertEqual(result["manifest_sha256"],sha((self.root/"publication-manifest.json").read_bytes()))

    def test_changed_missing_and_extra_files_rejected(self):
        original=self.write_manifest()
        (self.root/"data.bin").write_bytes(b"\x02\x01\x00")
        with self.assertRaisesRegex(PublicationError,"changed"):
            verify_manifest(self.root)
        (self.root/"data.bin").unlink()
        with self.assertRaisesRegex(PublicationError,"missing"):
            verify_manifest(self.root)
        (self.root/"data.bin").write_bytes(b"\x00\x01\x02")
        (self.root/"unlisted.txt").write_text("unexpected")
        with self.assertRaisesRegex(PublicationError,"extra"):
            verify_manifest(self.root)
        self.assertEqual(json.loads((self.root/"publication-manifest.json").read_text()),original)

    def test_unsafe_duplicate_and_self_entries_rejected(self):
        original=build_manifest(self.root)
        for name in ["../escape","/absolute","C:/drive","x\\escape","./README.md"]:
            broken=copy.deepcopy(original);broken["files"][0]["path"]=name;self.write_manifest(broken)
            with self.assertRaises(PublicationError):verify_manifest(self.root)
        broken=copy.deepcopy(original);broken["files"].append(copy.deepcopy(broken["files"][0]));self.write_manifest(broken)
        with self.assertRaisesRegex(PublicationError,"duplicate"):verify_manifest(self.root)
        broken=copy.deepcopy(original);broken["files"][0]["path"]="publication-manifest.json";self.write_manifest(broken)
        with self.assertRaisesRegex(PublicationError,"itself"):verify_manifest(self.root)
        broken=copy.deepcopy(original);broken["manifest_sha256"]="0"*64;self.write_manifest(broken)
        with self.assertRaisesRegex(PublicationError,"self-hash"):verify_manifest(self.root)

    def test_symlinks_and_case_collisions_rejected(self):
        (self.root/"link.bin").symlink_to(self.root/"data.bin")
        with self.assertRaisesRegex(PublicationError,"symlink"):build_manifest(self.root)
        (self.root/"link.bin").unlink()
        (self.root/"DATA.bin").write_bytes(b"duplicate on Windows")
        with self.assertRaisesRegex(PublicationError,"case-insensitive"):build_manifest(self.root)

    def test_protocol_sidecar_exact_and_canonical_hashes(self):
        protocol={"arms":["a"],"count":512}
        raw=(json.dumps(protocol,indent=2)+"\n").encode()
        (self.root/"protocol.json").write_bytes(raw)
        (self.root/"protocol.sha256").write_text(sha(raw)+"\n")
        manifest=self.write_manifest()
        self.assertEqual(manifest["identities"][0]["kind"],"file_sha256")
        verify_manifest(self.root)
        (self.root/"protocol.sha256").write_text(sha(canonical(protocol))+"\n")
        manifest=self.write_manifest()
        self.assertEqual(manifest["identities"][0]["kind"],"canonical_json_sha256")
        verify_manifest(self.root)
        (self.root/"protocol.sha256").write_text("0"*64+"\n")
        with self.assertRaisesRegex(PublicationError,"hash mismatch"):build_manifest(self.root)

    def test_result_and_correctness_reference_identity(self):
        folder=self.root/"TEST";folder.mkdir()
        (folder/"ARM.correctness.npz").write_bytes(b"retained fixture bytes")
        record={"correctness_artifact":{"file":"ARM.correctness.npz","sha256":sha((folder/"ARM.correctness.npz").read_bytes())}}
        raw=json.dumps(record).encode();(folder/"ARM.json").write_bytes(raw)
        index={"splits":{"TEST":{"ARM":{"file":"TEST/ARM.json","sha256":sha(raw)}}}}
        (self.root/"index.json").write_text(json.dumps(index))
        manifest=self.write_manifest()
        self.assertEqual(len(manifest["identities"]),2)
        verify_manifest(self.root)
        index["splits"]["TEST"]["ARM"]["sha256"]="0"*64
        (self.root/"index.json").write_text(json.dumps(index))
        with self.assertRaisesRegex(PublicationError,"hash mismatch"):build_manifest(self.root)

    def test_manifest_checksum_cycle_and_overwrite_rejected(self):
        self.write_manifest()
        with self.assertRaisesRegex(PublicationError,"never overwrites"):
            main(["--root",str(self.root),"--create"])
        (self.root/"publication-manifest.sha256").write_text(sha((self.root/"publication-manifest.json").read_bytes())+"\n")
        with self.assertRaisesRegex(PublicationError,"self-hash"):
            build_manifest(self.root)

    def test_canonical_v2_learned_manifest_identity_references(self):
        directory=self.root/"evaluation";directory.mkdir()
        manifest={"schema":"case010-learned-evaluation-v2-canonical-token-torch"}
        names={"protocol_sha256":"protocol.json","runtime_sha256":"evaluation_runtime.json",
               "calibration_sha256":"calibration.npz","token_table_config_sha256":"token_coefficients.json",
               "token_table_data_sha256":"token_coefficients.bin"}
        for field,name in names.items():
            raw=b"{}" if name.endswith(".json") else b"retained binary fixture"
            (directory/name).write_bytes(raw);manifest[field]=sha(raw)
        (directory/"manifest.json").write_text(json.dumps(manifest))
        publication=self.write_manifest()
        self.assertEqual(len(publication["identities"]),5)
        verify_manifest(self.root)
        (directory/"calibration.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(PublicationError,"hash mismatch"):
            build_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
