from pathlib import Path
import shutil
import pytest
from source.safe_run import completed
from source.contracts import dump
ROOT=Path(__file__).resolve().parents[1]
def base(tmp):
    shutil.copytree(ROOT/'configs',tmp/'configs');(tmp/'results').mkdir();return tmp

def test_completed_parity_is_read_only(tmp_path):
    r=base(tmp_path);shutil.copy2(ROOT/'results/parity.json',r/'results/parity.json');before=(r/'results/parity.json').read_bytes()
    assert completed(r,'parity') and (r/'results/parity.json').read_bytes()==before

def test_missing_parity_not_claimed_complete(tmp_path):assert not completed(base(tmp_path),'parity')
def test_failed_parity_not_overwritten(tmp_path):
    r=base(tmp_path);dump(r/'results/parity.json',{'rows':[{'loops':x,'pre_pil_bitwise':False,'official_generate_png_equal':False,'finite':True}for x in (1,2,4)]})
    with pytest.raises(ValueError,match='preserved'):completed(r,'parity')
def test_malformed_parity_not_overwritten(tmp_path):
    r=base(tmp_path);dump(r/'results/parity.json',{'rows':[]})
    with pytest.raises(ValueError,match='Malformed'):completed(r,'parity')
def test_existing_main_validated_without_models():assert completed(ROOT,'main')
