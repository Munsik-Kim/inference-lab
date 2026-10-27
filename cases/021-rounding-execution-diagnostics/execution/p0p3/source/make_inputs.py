"""Owned synthetic specifications, not external/user data or paraphrased sources.

Each document has a different subject, entity vocabulary and operating rules.
The common serializers are disclosed; documents are not independent real tasks.
The numerical scenario stream varies records within each authored specification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from common import CASE, write, file_row, byte_sha, utc

# Subject | purpose | named entities | domain-specific observations | decision rule
SPECS={
'explanation':[
('Water treatment shift','Explain how a small treatment plant keeps sediment out of a public supply', 'intake settling_tank sand_filter chlorine_store clearwell outlet', 'turbidity pressure flow chlorine contact_time filter_age', 'A turbidity increase closes the outlet until a second sample is checked. A clean pressure reading alone never overrides an unsafe water sample.'),
('Hill weather station','Explain why a station records observations before classifying a storm', 'ridge valley mast shelter gauge logger', 'wind_speed rainfall air_pressure humidity battery_voltage timestamp_delay', 'A storm alert requires a sustained wind threshold or a rapid pressure drop. Missing rain measurements remain unknown and are never converted to dry weather.'),
('Library conservation','Describe how conservators choose safe handling for fragile paper', 'atlas diary pamphlet map photograph ledger', 'paper_acidity binding_damage light_exposure moisture insect_marks handling_count', 'Items with loose binding are supported on a cradle. A readable page is not evidence that the binding can tolerate a flatbed scanner.'),
('Island ferry dispatch','Explain the relationship between a sailing window and passenger boarding', 'north_pier south_pier channel berth ramp terminal', 'tide_height crosswind visibility berth_depth boarding_count reserve_fuel', 'The dispatcher confirms channel depth before boarding begins. A full passenger list cannot justify departure outside the safe tide window.'),
('Seed bank rotation','Explain why seed viability and storage conditions are tracked separately', 'barley lentil millet rye bean sorghum', 'germination_rate moisture_content storage_months vial_seal freezer_temperature sample_mass', 'A low germination result triggers a confirmatory assay before regeneration. Opening a vial is logged because moisture exposure changes later storage decisions.'),
('Ambulance supply dispatch','Describe replenishment decisions for sealed emergency supply kits', 'depot ambulance clinic reserve_store inspection_desk dispatch_bay', 'seal_status expiry_days item_count route_minutes vehicle_load restock_delay', 'An expired sterile item is replaced even if the kit is otherwise full. The dispatch record separates medical readiness from vehicle travel time.'),
('Museum lighting plan','Explain how curators balance readability and protection from light', 'textile watercolor ceramic bronze manuscript display_case', 'illuminance exposure_hours visitor_distance reflection_count filter_age color_temperature', 'Sensitive paper receives a cumulative exposure limit. Raising room brightness is not an acceptable substitute for removing a reflection from a display label.'),
('Community microgrid','Explain dispatch decisions when a village balances solar power and storage', 'solar_roof battery inverter clinic_pump school_circuit backup_generator', 'generation demand state_of_charge inverter_temperature reserve_hours forecast_error', 'The controller preserves a minimum battery reserve for the clinic pump. An optimistic solar forecast cannot erase a measured storage deficit.')],
'key_value':[
('Backup restoration runbook','Record the recovery requirements of a local document archive', 'catalog media_index checksum_log nightly_copy offline_copy restore_station', 'retention_days verified_files unreadable_files restore_minutes snapshot_age available_space', 'A backup is usable only after a sample restore and checksum comparison. A successful copy command does not prove that the catalog can be recovered.'),
('Greenhouse ventilation sheet','Specify how a grower records ventilation decisions in separate growing zones', 'tomato_bay herb_bay seedling_bay east_vent west_vent roof_sensor', 'air_temperature leaf_temperature humidity vent_fraction soil_moisture fan_hours', 'Ventilation responds to measured leaf and air conditions. A dry sensor must be inspected before an irrigation cycle is added to the schedule.'),
('Equipment handover record','Track custody and inspection for portable survey instruments', 'tripod level prism rangefinder battery_case transport_box', 'serial_suffix calibration_days shock_count custody_days charge_percent inspection_minutes', 'The receiver checks the serial number and signs for physical condition separately. An intact transport box does not establish that an instrument remains calibrated.'),
('Rail inspection register','Define inspection notes for a short maintenance railway', 'switch crossing rail_joint sleeper drainage_culvert signal_post', 'gauge_error fastener_count wear_depth drainage_depth inspection_age defect_length', 'A gauge defect is escalated before reopening a track section. Replacing a missing fastener does not automatically resolve a neighboring alignment defect.'),
('Cold chain transfer record','Describe temperature evidence when sealed packages move between stores', 'loading_dock insulated_box transport_van receiving_desk freezer shelf', 'minimum_temperature maximum_temperature transit_minutes logger_gap seal_count remaining_hold_hours', 'A logger gap is reviewed as missing evidence, not a compliant temperature. Receiving staff record both package condition and the time the package reached cold storage.'),
('Battery cycling worksheet','Separate test conditions from acceptance decisions for rechargeable cells', 'cell_a cell_b cell_c reference_cell charger thermal_probe', 'capacity internal_resistance cycle_count rest_minutes surface_temperature cutoff_voltage', 'Capacity is compared at the same discharge rate and rest interval. A warm cell is allowed to cool before resistance measurements are accepted.'),
('Machine lubrication route','Record checks for an industrial maintenance route', 'gearbox bearing pump slideway reservoir return_filter', 'oil_level vibration particle_count running_hours oil_temperature leak_count', 'A particle increase triggers a sample before oil is replaced. Adding clean oil can dilute a symptom without identifying the component that produced the particles.'),
('Laboratory freezer audit','Define independent checks for cold storage and specimen identity', 'rack_one rack_two control_vial alarm_probe door_gasket backup_unit', 'temperature alarm_delay door_open_seconds vial_count frost_depth transfer_minutes', 'An alarm acknowledgement never substitutes for a temperature recovery record. A moved vial keeps its original identifier and receives a separate location history.')],
'code':[
('Inventory reservation algorithm','Implement deterministic allocation without creating stock that does not exist', 'washer bolt nut bracket gasket spring', 'available requested reserved damaged returned priority', 'Allocate at most the nonnegative usable stock. Preserve request order for equal priorities and record shortages instead of silently removing incomplete orders.'),
('Version routing algorithm','Select a supported protocol version for a client and a server', 'client_alpha client_beta gateway archive_node control_node fallback_node', 'minimum_version maximum_version preferred_version retry_count timeout_seconds handshake_age', 'Choose the highest common supported version. If the intersection is empty, return an explicit incompatibility result rather than assuming a legacy protocol works.'),
('Interval merge algorithm','Combine overlapping maintenance intervals while preserving uncovered gaps', 'inspection cleaning repair calibration cooling verification', 'start_minute end_minute crew_count travel_minutes setup_minutes interruption_count', 'Sort intervals by start and then end. Extend the active interval only when the next start is no later than the active end; otherwise emit the completed interval.'),
('Sliding counter algorithm','Maintain a bounded time window for monitoring a queue', 'arrival completion retry timeout cancel heartbeat', 'timestamp event_count window_seconds queue_depth worker_count sequence_number', 'Discard events strictly older than the left boundary and retain events exactly on it. Out of order timestamps must be rejected before the accumulator is changed.'),
('Configuration validation algorithm','Validate a small configuration before opening any service resources', 'listener storage logger scheduler monitor healthcheck', 'port retry_limit timeout_ms queue_limit batch_size retention_days', 'Collect every field error in a stable order. An unknown field is reported separately from a missing required field so that a misspelling cannot silently select a default.'),
('Tree traversal algorithm','Walk a directory-like tree while preventing accidental cycles', 'root drafts approved archive attachments exports', 'node_id parent_id child_count depth byte_count visit_order', 'Mark a node before visiting its children. Encountering an already active node signals a cycle; encountering a completed shared node can reuse its recorded summary.'),
('Ledger reconciliation algorithm','Pair invoices and payments without hiding duplicate identifiers', 'invoice payment credit_note refund adjustment deposit', 'record_id amount_cents posting_day account_id duplicate_count unmatched_count', 'Match on identifier and account before comparing integer cents. Keep unmatched rows on both sides and reject duplicate keys rather than choosing the last value.'),
('Stream frame parser','Decode length-prefixed messages from arbitrary input chunks', 'header payload checksum trailer buffer consumer', 'declared_length received_length checksum_value chunk_size message_count offset', 'Do not consume a partial header or payload. A length above the configured cap is an error before allocation; successful frames advance the cursor exactly once.')],
'json':[
('Field trip itinerary','Describe a school visit with separate transport and learning checkpoints', 'bus museum garden workshop lunch_point school_gate', 'arrival_minute departure_minute participant_count adult_count walking_distance capacity', 'Every student group has an assigned adult at each transfer. A later bus arrival changes the schedule explicitly and does not remove a safety check.'),
('Exhibition object manifest','Describe object movements and display prerequisites for a temporary exhibition', 'painting sculpture textile print model display_mount', 'object_number crate_number insured_value condition_score humidity_limit install_minutes', 'An object can be installed only after its condition is signed off and its mount is checked. A transport receipt is not an installation approval.'),
('Sensor calibration package','Record a calibration series with independent reference and device readings', 'reference_meter probe_one probe_two zero_source span_source chamber', 'reference_value observed_value tolerance dwell_seconds ambient_temperature repeat_number', 'A calibration point keeps both raw readings and the correction. Results outside tolerance are retained so the record does not become a list of successful points only.'),
('Warehouse inspection package','Describe warehouse aisle observations and follow-up assignments', 'aisle_a aisle_b loading_lane fire_exit pallet_rack charging_area', 'clear_width stack_height damaged_units occupancy_minutes inspection_round priority', 'Fire exits must remain clear even during loading. A temporary obstruction receives an owner and a removal deadline rather than a permanent exception flag.'),
('Film production schedule','Represent a small shoot with location permissions and equipment dependencies', 'courtyard studio staircase workshop cafe roof', 'scene_number setup_minutes take_count crew_count permit_end noise_limit', 'A location permit is checked before setup begins. An incomplete shot can move to a later slot only when continuity and required equipment are explicitly reviewed.'),
('Route closure notice','Represent a road maintenance closure and the routes that remain available', 'bridge junction depot school_turn service_lane detour', 'start_hour end_hour vehicle_limit lane_count detour_minutes notice_days', 'Emergency access is reviewed independently from general traffic. The detour notice states a start and an end time instead of assuming a closure lasts all day.'),
('Consent catalogue schema','Record permitted uses for a collection of invented demonstration assets', 'diagram audio_sample portrait_placeholder chart icon brochure', 'asset_number revision_number retention_days permission_count review_interval withdrawal_delay', 'Each allowed use is listed explicitly. A withdrawn permission blocks future distribution but the catalogue preserves the date and scope of the prior decision.'),
('Habitat census package','Represent field observations without treating absent observations as absent animals', 'wetland woodland meadow streambank reedbed rocky_slope', 'survey_minutes observer_count species_count visibility rainfall sample_area', 'A site with poor visibility is marked as limited coverage. Counts remain attached to survey effort so that a short visit is not compared as if it were a full census.')]
}

ALGORITHMS=[
'''def reserve(stock, requests):
    remaining = dict(stock)
    decisions = []
    ordered = sorted(enumerate(requests), key=lambda row: (-row[1]['priority'], row[0]))
    for original_index, request in ordered:
        item = request['item']
        amount = request['quantity']
        if amount < 0:
            raise ValueError('negative request')
        granted = min(amount, max(0, remaining.get(item, 0)))
        remaining[item] = remaining.get(item, 0) - granted
        decisions.append((original_index, item, granted, amount - granted))
    return sorted(decisions), remaining
''',
'''def negotiate(client_versions, server_versions, disabled=()):
    client = set(client_versions)
    server = set(server_versions)
    blocked = set(disabled)
    if any(not isinstance(v, int) or v < 1 for v in client | server):
        raise ValueError('invalid protocol version')
    common = sorted((client & server) - blocked)
    if not common:
        return {'status': 'incompatible', 'version': None}
    return {'status': 'selected', 'version': common[-1]}
''',
'''def merge_intervals(intervals):
    pending = sorted(intervals)
    merged = []
    for start, end in pending:
        if end < start:
            raise ValueError('reversed interval')
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
''',
'''from collections import deque
class WindowCounter:
    def __init__(self, width):
        if width <= 0:
            raise ValueError('nonpositive width')
        self.width, self.events, self.total = width, deque(), 0
        self.last_time = None
    def add(self, timestamp, count):
        if count < 0 or (self.last_time is not None and timestamp < self.last_time):
            raise ValueError('invalid event order or count')
        self.last_time = timestamp
        while self.events and self.events[0][0] < timestamp - self.width:
            self.total -= self.events.popleft()[1]
        self.events.append((timestamp, count))
        self.total += count
        return self.total
''',
'''def validate_config(config, limits):
    errors = []
    for name in sorted(set(config) - set(limits)):
        errors.append((name, 'unknown'))
    for name, (lower, upper) in sorted(limits.items()):
        if name not in config:
            errors.append((name, 'missing'))
            continue
        value = config[name]
        if type(value) is not int:
            errors.append((name, 'not integer'))
        elif not lower <= value <= upper:
            errors.append((name, 'outside inclusive limits'))
    return errors
''',
'''def postorder(graph, root):
    active, complete, result = set(), set(), []
    def visit(node):
        if node in active:
            raise ValueError('cycle')
        if node in complete:
            return
        active.add(node)
        for child in graph.get(node, []):
            visit(child)
        active.remove(node)
        complete.add(node)
        result.append(node)
    visit(root)
    return result
''',
'''def reconcile(invoices, payments):
    def index(rows):
        result = {}
        for row in rows:
            key = (row['record_id'], row['account_id'])
            if key in result:
                raise ValueError('duplicate key')
            result[key] = row['amount_cents']
        return result
    left, right = index(invoices), index(payments)
    return [{'key': key, 'invoice': left.get(key), 'payment': right.get(key),
             'matched': key in left and key in right and left[key] == right[key]}
            for key in sorted(set(left) | set(right))]
''',
'''class FrameParser:
    def __init__(self, cap):
        self.cap, self.buffer = cap, bytearray()
    def feed(self, chunk):
        self.buffer.extend(chunk)
        frames, cursor = [], 0
        while len(self.buffer) - cursor >= 4:
            length = int.from_bytes(self.buffer[cursor:cursor+4], 'big')
            if length > self.cap:
                raise ValueError('oversized frame')
            if len(self.buffer) - cursor < 4 + length:
                break
            frames.append(bytes(self.buffer[cursor+4:cursor+4+length]))
            cursor += 4 + length
        del self.buffer[:cursor]
        return frames
''']

def document(category,index,spec,rng):
    title,purpose,entities,fields,rule=spec
    entities,fields=entities.split(),fields.split()
    records=[]
    for i,entity in enumerate(entities):
        values={f:int(rng.integers(2,90)) for f in fields}
        records.append({'record_id':i+1,'entity':entity,'observations':values,
          'review_note':f'{entity} is inspected at its own checkpoint. Values for '+', '.join(fields[:3])+f' are recorded before the decision for this {title.lower()} is made.',
          'evidence_rule':rule if i==0 else f'The reviewer preserves the measured {fields[i%len(fields)]} and the original record identifier. A correction is appended with its reason; the earlier observation remains readable.',
          'next_action':f'Compare {fields[(i+1)%len(fields)]} with the applicable operating limit, record the result, and assign a follow-up for {entity} when evidence is incomplete.'})
    intro=f'{title}\nPurpose: {purpose}.\nThis is an invented example with no real people, facilities, or measurements. The numeric values are illustrative, not operating advice.\nOperating rule: {rule}\n'
    if category=='json':
        return json.dumps({'title':title,'purpose':purpose,'schema_version':1,'operating_rule':rule,'units':'illustrative dimensionless values; field names identify the intended quantity','records':records,'missing_value_policy':'Use null for an unobserved value. Never replace missing observations with zero. Keep the original record and append a dated correction.'},indent=2)
    if category=='key_value':
        lines=[intro,'record_format = sectioned key-value register','review_policy = retain original observations and append corrections']
        for record in records:
            lines.append('\n[record_'+str(record['record_id'])+']')
            for k,v in record.items():
                if isinstance(v,dict):lines.extend(f'{k}.{f} = {n}' for f,n in v.items())
                else:lines.append(f'{k} = {v}')
        return '\n'.join(lines)
    if category=='code':
        # Complete executable illustrative validation/decision input construction,
        # with a different authored algorithm specification for every document.
        lines=['"""'+intro+'\nImplementation review notes: Validate input before changing state. Preserve a stable ordering and return explicit errors. The examples below are synthetic fixtures, not production defaults.\n"""', 'from dataclasses import dataclass', '', '@dataclass(frozen=True)', 'class Observation:', '    entity: str', '    values: dict', '', 'def validate_record(record):', '    errors = []', '    if not record.entity:', '        errors.append("entity is required")', '    for name, value in sorted(record.values.items()):', '        if not isinstance(value, int) or value < 0:', '            errors.append((name, "expected nonnegative integer"))', '    return errors', '', 'def review_records(records):', '    reviewed = []', '    for sequence, record in enumerate(records):', '        errors = validate_record(record)', '        reviewed.append({"sequence": sequence, "entity": record.entity,', '                         "errors": errors, "accepted_for_review": not errors})', '    return reviewed', '', 'EXAMPLES = [']
        for r in records:
            lines.extend(['    # '+r['review_note'],'    # '+r['next_action'],f'    Observation({r["entity"]!r}, {r["observations"]!r}),'])
        lines.extend([']', '', 'if __name__ == "__main__":','    report = review_records(EXAMPLES)','    assert len(report) == len(EXAMPLES)','    assert [r["sequence"] for r in report] == list(range(len(EXAMPLES)))','    for row in report:','        print(row)'])
        lines.insert(1,ALGORITHMS[index])
        return '\n'.join(lines)
    paragraphs=[intro]
    for r in records:
        vals='; '.join(f'{k.replace("_"," ")} is {v}' for k,v in r['observations'].items())
        paragraphs.append(f'Checkpoint {r["record_id"]}: {r["entity"].replace("_"," ")}. {r["review_note"]} In the illustrative record, {vals}. {r["next_action"]}')
    paragraphs.append('The report separates an observation, a decision, and a follow-up. An observation records what was measured at a particular checkpoint. A decision applies the stated rule to that evidence. A follow-up identifies what must be checked next. Keeping these three fields separate makes uncertainty visible without pretending that an unobserved value was measured.')
    return '\n\n'.join(paragraphs)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
    dest=CASE/'inputs'; rows=[]
    for category_index,(category,specs) in enumerate(SPECS.items()):
        for i,spec in enumerate(specs):
            split='DEV' if i<2 else 'EVAL'; doc=f'{split.lower()}_{category}_{i:02d}'
            rng=np.random.default_rng(np.random.SeedSequence([61027021,category_index,i]))
            text=document(category,i,spec,rng); ids=np.asarray(tok.encode(text,add_special_tokens=False),np.int64)
            if ids.size<512:raise ValueError(f'TOO_SHORT:{doc}:{ids.size}')
            textpath=dest/(doc+'.txt');textpath.write_text(text+'\n',encoding='utf8')
            np.save(dest/(doc+'.npy'),ids[:512],allow_pickle=False)
            rows.append({'document_id':doc,'split':split,'category':category,'title':spec[0],'total_source_tokens':ids.size,'text':file_row(textpath),'ids':file_row(dest/(doc+'.npy')),'ids_bytes_sha256':byte_sha(ids[:512].tobytes()),'seed_sequence':[61027021,category_index,i]})
    if len({r['ids_bytes_sha256'] for r in rows})!=32:raise ValueError('DUPLICATE_INPUT')
    order=np.random.default_rng(61027022)
    splitorders={s:[rows[int(i)]['document_id'] for i in order.permutation([i for i,r in enumerate(rows) if r['split']==s])] for s in ('DEV','EVAL')}
    conditions=[f'{d}_{r}' for d in ('D0','D1','P') for r in ('R0','R1')]
    conditionorder=list(order.permutation(conditions))
    write(dest/'manifest.json',{'created_utc':utc(),'root_seed':61027021,'execution_seed':61027022,'tokenizer_revision':'c1899de289a04d12100db370d81485cdf75e47ca','special_tokens':False,'chat_template':False,'mask':'ones, int64','position_ids':'arange(N), int64','lengths':[128,512],'selected_positions':{str(n):[((j+1)*n)//17 for j in range(16)] for n in (128,512)},'source_claim':'32 independently authored subject specifications serialized with common category templates; synthetic diagnostic corpus, not independent real user workloads','documents':rows,'document_order':splitorders,'condition_order':conditionorder,'repeat_documents':splitorders['DEV'][:3]})
    print(json.dumps({'documents':len(rows),'min_tokens':min(r['total_source_tokens'] for r in rows),'condition_order':conditionorder}))

if __name__=='__main__':main()
