"""Create fixed, self-authored confirmation inputs; never downloads weights."""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import CASE,sha,write_json,fingerprint,utc


def document(i,language):
    title=f'Case 004 fictional archive CONF-{i:02d}: operator audit source text. This is synthetic, not a customer record.\n'
    paragraphs=[title]
    for j in range(85):
        number=10000+i*127+j*19
        if language=='en':
            templates=[
              f'Archive entry {j}: station Kestrel-{i} recorded parcel {number}. The blue container was inspected by the morning team; the evening team checked its routing label and recorded a separate receipt. The note distinguishes a proposed change from an accepted instruction.\n',
              f'In section {j}, the fictional library maps shelf {number%31} to catalog item {number}. Readers may request a copy through the internal desk. A correction preserves the earlier author field while replacing a mistaken publication month. No real institution or customer is described.\n',
              f'Observation {j} compares two draft procedures for project Harbor-{i}. The first checks every batch on arrival; the second schedules a weekly inventory. The committee retained the arrival check and documented why the suggested weekly interval was unsuitable for fragile items.\n']
        elif language=='ko':
            templates=[
              f'가상 기록 {j}번: 해오름-{i} 창고의 묶음 {number}를 오전에 검사했다. 담당자는 상자 색상과 인수 표식을 따로 기록했다. 오후에 제안된 이동 일정은 검토 중이며, 기존에 확정된 도착 장소를 변경한 것은 아니다. 실제 고객이나 기관의 자료가 아니다.\n',
              f'문서 항목 {j}에는 가상 도서관의 서가 {number%31}와 목록 번호 {number}가 연결되어 있다. 첫 작성자는 표지의 연도를 잘못 옮겼고 후속 기록에서 그 연도만 고쳤다. 제목과 책임자 표기는 이전 기록을 유지한다.\n',
              f'운영 검토 {j}에서 항구-{i} 팀은 매일 점검과 주간 점검을 비교했다. 취급 물품의 상태 변화가 빨라 매일 점검을 유지했다. 점검 횟수 자체가 정확성을 보장하지 않으므로 담당자가 근거와 예외를 함께 기록하도록 했다.\n']
        else:
            templates=[
              f'# Synthetic ledger {i}, rule {j}; no production data\ndef route_{i}_{j}(record):\n    accepted = record.get("revision", 0) >= {j%7}\n    bucket = (record["sequence"] + {number}) % 31\n    return {{"accepted": accepted, "bucket": bucket, "reason": "stable key order"}}\n',
              f'# Example invariant {j}: proposed and committed revisions remain separate.\nentry_{i}_{j} = {{"id": {number}, "status": "committed", "label": "synthetic"}}\nassert entry_{i}_{j}["status"] != "proposed"\n# Equality checks cover values; iteration order is not a timestamp.\n',
              f'# Archive {i} test vector {j}\ndef merge_{i}_{j}(old, patch):\n    result = dict(old)\n    for key, value in patch.items():\n        if key not in ("identity", "created_at"):\n            result[key] = value\n    return result\n# Inputs are illustrative mappings, not private records.\n']
        paragraphs.append(templates[(j+i)%3])
    return '\n'.join(paragraphs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case003',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);args=p.parse_args()
    if (CASE/'configs/experiment_spec.json').exists():raise RuntimeError('Do not regenerate frozen inputs')
    from transformers import AutoTokenizer
    model=json.loads((args.case003/'provenance/model.json').read_text())
    assert args.snapshot.name==model['revision']
    checked=[]
    for item in model['files']:
        path=args.snapshot/item['name'];assert path.is_file(),'Pinned snapshot absent; no automatic download'
        assert sha(path)==item['verified_sha256'],item['name'];checked.append(item)
    tok=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True,trust_remote_code=False)
    old=json.loads((args.case003/'inputs/manifest.json').read_text())['documents'];docs=[]
    for i,d in enumerate(x for x in old if x['split']=='dev'):
        docs.append((f'c004-dev-{i:02d}','dev',d['language'],(args.case003/'inputs'/(d['document_id']+'.txt')).read_text(),d['document_id']))
    languages=['en']*5+['ko']*5+['code']*6
    for i,language in enumerate(languages):docs.append((f'c004-confirm-{i:02d}','confirmation',language,document(i,language),None))
    manifest=[]
    for did,split,lang,text,oldid in docs:
        ids=tok.encode(text,add_special_tokens=False);assert len(ids)>=4096,(did,len(ids))
        path=CASE/'inputs'/(did+'.txt');path.write_text(text)
        prefixes={str(n):fingerprint(ids[:n]) for n in [512,2048,4096]}
        manifest.append(dict(document_id=did,split=split,language=lang,source_case003_dev_id=oldid,text_file='inputs/'+path.name,text_sha256=sha(path),token_ids_4096=ids[:4096],available_tokens=len(ids),prefix_sha256=prefixes))
    assert len(set(x['text_sha256'] for x in manifest))==24
    assert len(set(x['prefix_sha256']['512'] for x in manifest))==24
    oldprefixes={hashlib.sha256(json.dumps(d['token_ids_4096'][:512],sort_keys=True,separators=(',',':')).encode()).hexdigest() for d in old}
    assert not oldprefixes & {d['prefix_sha256']['512'] for d in manifest if d['split']=='confirmation'}
    write_json(CASE/'inputs/manifest.json',dict(created_at_utc=utc(),model_id=model['model_id'],revision=model['revision'],tokenizer_revision=model['revision'],encoding='plain text, no chat template, no added special tokens',source='Eight Case003 development texts reused read-only; sixteen new self-authored synthetic confirmation documents',prefix_dependence='512/2048/4096 share one document prefix; document is the independent cluster',documents=manifest))
    write_json(CASE/'provenance/model.json',dict(model_id=model['model_id'],revision=model['revision'],files=checked,cache_status='EXISTING_VERIFIED_NO_DOWNLOAD',config=json.loads((args.snapshot/'config.json').read_text())))
    print(json.dumps({'documents':len(manifest),'dev':8,'confirmation':16,'model_files_verified':len(checked),'minimum_document_tokens':min(x['available_tokens'] for x in manifest)}))
if __name__=='__main__':main()
