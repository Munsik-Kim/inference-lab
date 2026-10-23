"""Publication comparison contract, fixed before clean-copy observations."""
import math,json,csv
from pathlib import Path

NUMERIC_ATOL=1e-10
NUMERIC_RTOL=1e-9

def compare_tree(actual,expected,path='$'):
    """Discrete values exact; finite derived floats use the declared CSV rule.

    Scientific raw/projector/reference gates still use each frozen protocol.
    This rule handles aggregation/BLAS portability, not model tolerances.
    """
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('KEYS '+path)
        for k in expected:compare_tree(actual[k],expected[k],path+'.'+k)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise ValueError('LENGTH '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare_tree(a,b,f'{path}[{i}]')
    elif isinstance(expected,float):
        if not isinstance(actual,(int,float)) or not math.isfinite(actual) or not math.isclose(actual,expected,abs_tol=NUMERIC_ATOL,rel_tol=NUMERIC_RTOL):
            raise ValueError('DERIVED_FLOAT '+path+f' {actual} vs {expected}')
    elif type(actual)!=type(expected) or actual!=expected:
        raise ValueError('DISCRETE '+path)

def compare_csv(actual,expected):
    with Path(actual).open() as f:a=list(csv.reader(f))
    with Path(expected).open() as f:b=list(csv.reader(f))
    if len(a)!=len(b):raise ValueError('CSV_ROWS')
    count=0
    for aa,bb in zip(a,b):
        if len(aa)!=len(bb):raise ValueError('CSV_COLUMNS')
        for x,y in zip(aa,bb):
            if x==y:continue
            try:
                xx,yy=json.loads(x),json.loads(y)
            except (json.JSONDecodeError,ValueError):raise ValueError('CSV_TEXT_DIFFERENCE')
            compare_tree(xx,yy);count+=1
    return {'cells_with_numerically_equivalent_text':count,'rows':len(a)}
