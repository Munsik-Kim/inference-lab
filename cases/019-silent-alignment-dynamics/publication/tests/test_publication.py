"""Publication boundaries, safe restoration and portability, no scientific ODE."""
import sys,json,tempfile,tarfile,io,hashlib,unittest,copy
from pathlib import Path
PUB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PUB))
from verify import safe,validate_mapping,validate_distribution,restore,verify_restored,sha,canonical
from portable import compare_tree

class Publication(unittest.TestCase):
    def fixture(self,root,kind='regular',duplicate=False):
        c=root/'case';(c/'publication/provenance').mkdir(parents=True)
        payload=b'exact record bytes\n';h=hashlib.sha256(payload).hexdigest();member='objects/'+h
        archive=root/'evidence.tar.xz'
        with tarfile.open(archive,'w:xz') as t:
            info=tarfile.TarInfo(member)
            if kind=='regular':info.size=len(payload);t.addfile(info,io.BytesIO(payload))
            else:
                info.type={'symlink':tarfile.SYMTYPE,'hardlink':tarfile.LNKTYPE,'directory':tarfile.DIRTYPE,'device':tarfile.CHRTYPE}[kind];info.linkname='../outside';t.addfile(info)
            if duplicate:
                info=tarfile.TarInfo(member);info.size=len(payload);t.addfile(info,io.BytesIO(payload))
        row={'path':'case/raw.txt','member':member,'sha256':h,'bytes':len(payload),'retention':'EXACT_BYTES','transforms':[]}
        a={'bytes':archive.stat().st_size,'sha256':sha(archive),'logical_files':2,'unique_objects':1,'restore_bytes':2*len(payload),'unique_uncompressed_bytes':len(payload),'max_object_bytes':len(payload)}
        m={'archive':a,'files':[row,dict(row,path='case/alias.txt')]}
        (c/'publication/provenance/original_mapping.json').write_text(json.dumps(m));return c,archive,m

    def test_relative_path_rejects_traversal_absolute_and_windows(self):
        for x in ['/tmp/x','../x','a/../x','a//b','a/./b','C:/x','a\\b','a\x00b']:
            with self.assertRaises(ValueError,msg=x):safe(x)
        self.assertEqual(str(safe('case/results/raw.npz')),'case/results/raw.npz')

    def test_exact_regular_restore_with_hash_aliases(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,a,m=self.fixture(p);r=restore(c,a,p/'new')
            self.assertEqual(r['logical_files'],2);self.assertEqual((p/'new/case/raw.txt').read_bytes(),(p/'new/case/alias.txt').read_bytes())

    def test_duplicate_member_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,a,m=self.fixture(p,duplicate=True)
            with self.assertRaises(ValueError):restore(c,a,p/'new')

    def test_links_directories_special_files_rejected(self):
        for kind in ['symlink','hardlink','directory','device']:
            with tempfile.TemporaryDirectory() as d:
                p=Path(d);c,a,m=self.fixture(p,kind=kind)
                with self.assertRaises(ValueError):restore(c,a,p/'new')

    def test_alias_cycle_and_inconsistent_alias_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,a,m=self.fixture(p);m['files'][0]['alias_of']='case/alias.txt'
            with self.assertRaises(ValueError):validate_mapping(m)
            del m['files'][0]['alias_of'];m['files'][1]['bytes']+=1
            with self.assertRaises(ValueError):validate_mapping(m)

    def test_existing_output_and_output_inside_case_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,a,m=self.fixture(p)
            with self.assertRaises(ValueError):restore(c,a,p)
            with self.assertRaises(ValueError):restore(c,a,c/'new')

    def test_archive_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,a,m=self.fixture(p);b=bytearray(a.read_bytes());b[40]^=1;a.write_bytes(b)
            with self.assertRaises(ValueError):restore(c,a,p/'new')

    def test_restored_directory_reuse_checks_all_bytes_and_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,a,m=self.fixture(p);restore(c,a,p/'new')
            self.assertEqual(verify_restored(c,p/'new')['logical_files'],2)
            (p/'new/case/raw.txt').write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_restored(c,p/'new')

    def test_xz_crc_checked_even_if_tar_end_was_read(self):
        import lzma
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);c,a,m=self.fixture(p);b=bytearray(a.read_bytes());b[-10]^=1;a.write_bytes(b)
            m['archive']['sha256']=sha(a);(c/'publication/provenance/original_mapping.json').write_text(json.dumps(m))
            with self.assertRaises((lzma.LZMAError,EOFError,tarfile.ReadError)):restore(c,a,p/'new')

    def test_numeric_policy_keeps_discrete_status_and_null_exact(self):
        compare_tree({'x':1.+1e-12,'status':'NOT_REACHED','time':None},{'x':1.,'status':'NOT_REACHED','time':None})
        with self.assertRaises(ValueError):compare_tree({'time':0.},{'time':None})
        with self.assertRaises(ValueError):compare_tree({'x':1.01},{'x':1.})

    def test_release_location_is_explicit_and_checks_do_not_download(self):
        case=PUB.parent;dist=json.loads((PUB/'distribution.json').read_text())
        self.assertEqual(dist['strategy'],'B')
        self.assertEqual(dist['asset']['tag'],'case019-evidence-v1.0.0')
        self.assertEqual(dist['asset']['url'],'https://github.com/Munsik-Kim/inference-lab/releases/download/case019-evidence-v1.0.0/case019-recorded-evidence-v1.tar.xz')
        source=(PUB/'run_checks.py').read_text();self.assertNotIn('urlopen',source);self.assertNotIn('requests.get',source)
        self.assertEqual(json.loads((PUB/'quick/SLICE.json').read_text())['status'],'SLICE')

    def test_release_location_cannot_change_archive_identity(self):
        mapping=json.loads((PUB/'provenance/original_mapping.json').read_text());dist=json.loads((PUB/'distribution.json').read_text())
        validate_distribution(mapping,dist)
        for key in ['bytes','sha256','restore_bytes']:
            bad=copy.deepcopy(dist);bad['asset'][key]='changed'
            with self.assertRaisesRegex(ValueError,'EVIDENCE_IDENTITY_CHANGED'):validate_distribution(mapping,bad)

    def test_release_url_rejects_wrong_repository_scheme_and_tag(self):
        mapping=json.loads((PUB/'provenance/original_mapping.json').read_text());dist=json.loads((PUB/'distribution.json').read_text())
        for url in ['file:///tmp/evidence','https://example.invalid/evidence',dist['asset']['url'].replace('Munsik-Kim','other')]:
            bad=copy.deepcopy(dist);bad['asset']['url']=url
            with self.assertRaisesRegex(ValueError,'CASE019_RELEASE_URL'):validate_distribution(mapping,bad)
        bad=copy.deepcopy(dist);bad['asset']['tag']='../main'
        with self.assertRaisesRegex(ValueError,'CASE019_RELEASE_TAG'):validate_distribution(mapping,bad)

if __name__=='__main__':unittest.main()
