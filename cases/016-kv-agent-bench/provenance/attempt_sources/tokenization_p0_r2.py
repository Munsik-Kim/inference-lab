"""Lazy, offline tokenizer preparation; core modules do not import ML packages."""
import copy
import hashlib
from .fixture_generator import SYSTEM,message_parts,tools
from .common import digest

def get_tokenizer(snapshot):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)

def render(tok,messages,tool_list):
    return tok.apply_chat_template(messages,tools=tool_list,tokenize=False,add_generation_prompt=True,enable_thinking=False)

def token_hash(ids):
    # Explicit byte representation, never Python's randomized hash().
    import struct
    return hashlib.sha256(b''.join(struct.pack('<I',i) for i in ids)).hexdigest()

def prepare_input(tok,task,menu,length):
    parts=message_parts(task);padding=['','','',''];anchors=([3600,5500,7350] if length==8192 else [6500,10500,15000])
    def msg():return [{'role':'system','content':SYSTEM},{'role':'user','content':''.join(padding[i]+parts[i] for i in range(4))}]
    def text():return render(tok,msg(),tools(menu))
    def position(i):
        x=text();at=x.index(parts[i]);return len(tok.encode(x[:at],add_special_tokens=False))
    for i,goal in enumerate(anchors):
        for _ in range(8):
            diff=goal-position(i)
            if diff<0:raise ValueError(f'tool/prefix exceeds preregistered evidence anchor: {task["task_id"]} {menu} {i}')
            if diff==0:break
            padding[i]+=' archive'*diff
        if position(i)!=goal:raise ValueError('could not exactly align evidence')
    for _ in range(8):
        ids=tok.encode(text(),add_special_tokens=False);diff=length-len(ids)
        if diff<0:raise ValueError('registered input length too small')
        if diff==0:break
        padding[3]+=' archive'*diff
    messages=msg();rendered=text();ids=tok.encode(rendered,add_special_tokens=False)
    if len(ids)!=length:raise ValueError('exact first prompt length mismatch')
    return {'task_id':task['task_id'],'menu':menu,'length':length,'messages':messages,'tools':tools(menu),
      'input_token_ids':ids,'input_token_sha256':token_hash(ids),'rendered_prompt_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
      'surface_hash':digest({'messages':messages,'tools':tools(menu)}),'evidence_token_offsets':{k:position(i) for i,k in enumerate(task['essential_facts'])},
      'add_special_tokens':False,'enable_thinking':False,'length_includes_tools_and_template':True}
