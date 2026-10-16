# A1 protocol

See [frozen protocol](specs/protocol.json), [schemas](data/schema.json), [mapping](data/api_mappings.json), [scenarios](data/scenarios.json), [oracle](data/oracle.json), [input identities](provenance/input_manifest.json), and [source freeze](provenance/source_freeze.json). One response has one immutable available-result list. Correct write→validate in one response is allowed; read→multiply requires receipt in a later generation. Final validation must match the final state version.
