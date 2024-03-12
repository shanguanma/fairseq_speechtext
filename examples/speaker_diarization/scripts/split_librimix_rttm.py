gemeng_dev_rttm = '/opt/tiger/unit_fairseq/examples/decoder_only/alimeeting3/librimix/dev_100.scp'

dev_utt = set()
with open(gemeng_dev_rttm) as f:
    for line in f:
        dev_utt.add(line.strip('\n').split(' ')[1].split('/')[-1][:-4])

ori_train_rttm = '/opt/tiger/unit_fairseq/examples/decoder_only/alimeeting3/librimix/rttm_train'
new_dev_rttm = []
new_train_rttm = []

with open(ori_train_rttm) as f:
    for line in f:
        if line.split('\t')[1] in dev_utt:
            new_dev_rttm.append(line)
        else:
            new_train_rttm.append(line)

with open('/opt/tiger/unit_fairseq/examples/decoder_only/alimeeting3/librimix/rttm_train_new', 'w') as f:
    f.write(''.join(new_train_rttm))

with open('/opt/tiger/unit_fairseq/examples/decoder_only/alimeeting3/librimix/rttm_dev_new', 'w') as f:
    f.write(''.join(new_dev_rttm))
