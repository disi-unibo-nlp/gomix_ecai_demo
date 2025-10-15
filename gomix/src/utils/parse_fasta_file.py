def parse_fasta_file(fasta_file: str):
    with open(fasta_file) as f:
        seq_id = None
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_id is not None:
                    yield {'seq_id': seq_id, 'seq': seq}
                seq_id = line[1:]
                seq = ""
            else:
                seq += line
        if seq_id is not None:
            yield {'seq_id': seq_id, 'seq': seq}
