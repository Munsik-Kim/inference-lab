"""Deterministic synthetic facts, Korean documents, and independent gold."""
import argparse, datetime, hashlib, json, pathlib, random

FIELDS = ['owner', 'task', 'due_date', 'amount_krw']
BASE_PROMPT = '''문서에서 대상 요청의 현재 확정 정보만 추출하세요. 확정 변경은 해당 필드만 갱신하며, 제안·철회된 변경과 다른 요청은 무시합니다. owner와 task는 문서의 담당자·작업명 표기 그대로, due_date는 YYYY-MM-DD, amount_krw는 원 단위 정수로 쓰세요. 미기재 정보는 null입니다. 설명이나 코드 블록 없이 정확히 owner, task, due_date, amount_krw 네 키의 JSON 객체만 출력하세요.'''
PROMPT = '''대상 ID와 정확히 일치하는 기록만 사용하세요. 확정 변경은 해당 필드만 갱신하고 제안·철회는 무시합니다. 다른 요청의 값으로 보충하지 마세요. owner와 task는 담당자·작업명 원문 그대로, due_date는 YYYY-MM-DD, amount_krw는 원 단위 정수입니다. 미기재는 null입니다. 설명·코드 블록 없이 owner, task, due_date, amount_krw 네 키의 JSON 객체만 출력하세요.'''
NAMES = ['김민수','이서연','박지훈','최유진','정다은','한도윤','윤서현','장하준','오수빈','서지우','백예린','임태호']
TASKS = ['서버 점검','계약서 검토','회의실 배선 정리','재고 대장 정비','보안 교육 자료 제작','장비 반납 확인','행사 안내문 발송','백업 복원 점검','번역본 교정','설문 응답 집계','전력 사용량 조사','출입 카드 교체','배송 일정 확인','웹 접근성 점검','문서 보존 목록 작성','사무용품 구매','좌석 배치도 수정','운영 매뉴얼 교정']
LABELS = dict(owner='담당자', task='작업명', due_date='마감일', amount_krw='금액')

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))

def read_jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]

def messages(doc):
    # This exact allowlist prevents gold, evidence, and scenario metadata leakage.
    return [{'role':'user','content':PROMPT+'\n문서:\n'+doc['document']+'\n대상 요청 ID: '+doc['target_id']}]

def construction_messages(doc):
    # Retain original length construction so the one dev-only prompt revision
    # does not alter any document, fact, event, or gold value.
    return [{'role':'user','content':BASE_PROMPT+'\n대상 요청 ID: '+doc['target_id']+'\n문서:\n'+doc['document']}]

def value(field, rng):
    if field=='owner': return rng.choice(NAMES)
    if field=='task': return rng.choice(TASKS)
    if field=='due_date': return f'2026-{rng.choice([10,11,12]):02d}-{rng.randint(1,28):02d}'
    return rng.choice([0,35000,78000,125000,250000,480000,720000,1350000,2480000])

def render_value(field, v, style):
    if v is None: return '미기재'
    if field=='due_date':
        y,m,d=map(int,v.split('-'))
        return [v,f'{y}년 {m}월 {d}일',f'{y}.{m:02d}.{d:02d}'][style%3]
    if field=='amount_krw':
        korean=(f'{v//10000}만 {v%10000:,}원' if v%10000 else f'{v//10000}만 원') if v>=10000 else f'{v}원'
        return [f'{v:,}원',korean,f'KRW {v:,}'][style%3]
    return v

def record_line(rid, status, patch, rng):
    fields=list(patch);rng.shuffle(fields)
    return f'{rid} [{status}] '+ ' / '.join(LABELS[f]+': '+render_value(f,patch[f],rng.randrange(3)) for f in fields)

def make_fact(i, split, bucket, rng):
    cid=f'{split}-{bucket}-{i:03d}'
    rid=f'RQ-{10000+i+(0 if split=="dev" else 20000)}'
    # Balanced missingness is assigned before rendering or model execution.
    null_fields=[[],['owner'],['task'],['due_date'],['amount_krw'],[],[],['owner','amount_krw'],[],[]][i%10]
    initial={f:None if f in null_fields else value(f,rng) for f in FIELDS}
    events=[]
    allowed=[f for f in FIELDS if f not in null_fields]
    for j in range(i%3):
        f=allowed[(i+j)%len(allowed)]; v=value(f,rng)
        while v==initial[f]: v=value(f,rng)
        events.append({'status':'confirmed','version':j+2,'patch':{f:v}})
    f=FIELDS[i%4]
    events.append({'status':'proposed' if i%2 else 'withdrawn','version':9,'patch':{f:value(f,rng)}})
    return {'id':cid,'scenario_id':f'scenario-{split}-{i:03d}','target_id':rid,'split':split,'bucket':bucket,'initial':initial,'events':events,'target_active':True}

def side_paragraph(index, rng, long=False):
    rid=f'OTHER-{index:04d}'
    name=rng.choice(NAMES);task=rng.choice(TASKS);date=render_value('due_date',value('due_date',rng),index)
    amount=render_value('amount_krw',value('amount_krw',rng),index+1)
    patterns=[
        f'{rid} 접수 메모: {name} 담당의 {task} 건은 {date} 마감, 확정 금액 {amount}이다.',
        f'별건 {rid} 검토 결과 — 작업명 {task}, 금액 {amount}, 담당자 {name}, 마감 {date}.',
        f'회의 안건 {rid}: 작업 「{task}」의 담당은 {name}이다. {date}까지 제출한다. 승인 예산: {amount}.',
        f'{rid} 이메일 요약: 작업 「{task}」의 완료일은 {date}이다. 결재액은 {amount}, 실무 담당은 {name}이다.',
        f'취소 요청 {rid}: {name}의 「{task}」 요청은 취소됐다. 기존 {date} 일정과 {amount} 집행은 중단한다.',
        f'{rid} 인수인계표 | 담당 {name} | 작업 {task} | 금액 {amount} | 마감 {date}.',
        f'별도 승인서 {rid}의 기재 사항은 다음과 같다. 작업: {task}, 비용: {amount}, 담당자: {name}. 확정 마감일: {date}.',
        f'{rid} 변경 제안: 작업 「{task}」의 조정안(담당자: {name}, 마감일: {date}, 금액: {amount})은 미승인 상태다.',
    ]
    extra=[
        '첨부 대장의 순번은 접수 순서일 뿐 우선순위를 뜻하지 않는다.',
        '작업 결과는 지정된 검수 항목과 대조한 후 별도 보관한다.',
        '재무팀은 집행 전 증빙의 발행일과 청구 주체를 다시 대조한다.',
        '회의 참석자는 해당 건의 담당 변경을 추가로 의결하지 않았다.',
        '시설팀은 현장 출입 시간을 관리대장에 남기기로 했다.',
        '구매 담당 부서는 수량 단위와 포장 조건을 공급 명세서와 확인한다.',
        '초안의 맞춤법과 표 번호는 편집 담당자가 점검한다.',
        '작업 완료 통지는 내부 게시판의 해당 요청 항목에 남긴다.',
        '검수자는 미완료 항목을 결과물의 별도 비고란에 적는다.',
        '견적 파일과 최종 승인서는 구분해 보관한다.',
        '보안팀은 공유 자료에 실제 고객 식별자가 없는지 확인한다.',
        '제출된 결과물에는 작성일과 검토 범위를 표시한다.',
    ]
    # No repeated padding sentences: each supplementary sentence appears at most once.
    local_index=index%100
    return patterns[index%len(patterns)]+(' '+extra[local_index] if long and local_index<len(extra) else '')

def render(fact, rng, tokenizer):
    rid=fact['target_id']
    initial_line=record_line(rid,'접수·확정 v1',fact['initial'],rng)
    lines=[initial_line]
    gold=dict(fact['initial']);evidence={f:initial_line for f in FIELDS}
    for e in fact['events']:
        status={'confirmed':'확정 변경','proposed':'변경 제안·미승인','withdrawn':'변경안 철회·적용 안 함'}[e['status']]
        line=record_line(rid,status+f' v{e["version"]}',e['patch'],rng)
        lines.append(line)
        if e['status']=='confirmed':
            gold.update(e['patch']);evidence.update({f:line for f in e['patch']})
    # Explicit versions remove ambiguity even when documents reorder entries.
    rng.shuffle(lines)
    header=rng.choice(['업무 요청 기록. 확정 변경은 버전 순으로 적용한다.','업무 인수인계 발췌. 높은 버전의 확정 변경이 우선한다.','승인 대장과 변경 기록. 확정 버전 순으로 현재 값을 정한다.'])
    doc={k:fact[k] for k in ['id','split','bucket','target_id']}
    def count(ls):
        doc['document']=header+'\n'+'\n'.join(ls)
        return len(tokenizer.apply_chat_template(construction_messages(doc),tokenize=True,add_generation_prompt=True,return_dict=False))
    goal=512 if fact['bucket']=='short' else 4096
    lower,upper=int(goal*.98),int(goal*1.1)
    n=count(lines)
    if n>upper: raise ValueError(('base too long',fact['id'],n))
    side=0
    while n<lower:
        # Each distractor has distinct facts and ID; target evidence is never cut.
        s=side_paragraph(side+int(fact['id'].split('-')[-1])*100,rng,fact['bucket']=='long')
        candidate=lines.copy();candidate.insert(rng.randrange(len(lines)+1),s)
        new=count(candidate)
        if new>upper:
            s=f'관리 메모 {side+1}: 접수 번호는 다른 요청과 합쳐 쓰지 않는다.'
            candidate=lines+[s];new=count(candidate)
        if new>upper: raise ValueError(('cannot fit',fact['id'],n,new))
        lines=candidate;n=new;side+=1
        if side>200: raise RuntimeError('length construction failed')
    count(lines)
    doc['input_tokens']=len(tokenizer.apply_chat_template(messages(doc),tokenize=True,add_generation_prompt=True,return_dict=False))
    assert goal*.9<=doc['input_tokens']<=goal*1.1
    assert all(q in doc['document'] for q in evidence.values())
    return doc,{'id':fact['id'],'values':gold,'evidence':evidence}

def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    rng=random.Random(20260910);allfacts=[];stats={};allids=set();scenarios=set()
    for split,n in [('dev',20),('eval',100)]:
        docs=[];golds=[]
        for i in range(n):
            bucket='short' if i<n//2 else 'long'
            fact=make_fact(i,split,bucket,rng);doc,gold=render(fact,rng,tok)
            assert fact['id'] not in allids and fact['scenario_id'] not in scenarios
            allids.add(fact['id']);scenarios.add(fact['scenario_id']);allfacts.append(fact);docs.append(doc);golds.append(gold)
            # Recompute gold from facts, independent of rendering.
            expected=dict(fact['initial'])
            for event in fact['events']:
                if event['status']=='confirmed':expected.update(event['patch'])
            assert expected==gold['values']
            assert messages(doc)==messages({**doc,'values':gold['values'],'evidence':gold['evidence'],'scenario_id':fact['scenario_id']})
        jsonl(a.root/'data'/f'{split}_inputs.jsonl',docs);jsonl(a.root/'data'/f'{split}_gold.jsonl',golds)
        for bucket in ['short','long']:
            counts=[d['input_tokens'] for d in docs if d['bucket']==bucket]
            stats[f'{split}_{bucket}']={'n':len(counts),'min':min(counts),'max':max(counts),'mean':sum(counts)/len(counts)}
    jsonl(a.root/'data/facts.jsonl',allfacts)
    dump(a.root/'configs/prompt.json',{'version':2,'instruction':PROMPT,'role':'user','target_id_position':'after document','adjustments_after_dev':1})
    dump(a.root/'data/manifest.json',{'version':2,'facts_version':1,'prompt_version':2,'seed':20260910,'synthetic':True,'human_review':False,'construction':'Structured facts and confirmed versioned events determine gold before model execution. Rendering is deterministic. No model output or LLM judge determines gold. Shared generation grammar across splits. Documents/gold unchanged after the one dev-only prompt revision; input-token metadata recalculated.','normalization':'Unicode NFC and strip on strings only','length_includes':'user instruction, target ID, document, pinned chat template and generation prompt','token_counts':stats,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((a.root/'data').glob('*.jsonl'))}})
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
