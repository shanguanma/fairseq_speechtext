import subprocess

rttm_path = (
    "/workspace/junyi/codebase/joint-optimization/SCTK-2.4.12/libri23mix_max.rttm"
)

segments = {}
max_len = {}
overlap_segs = {}
non_overlap_segs = {}
with open(rttm_path) as f:
    for line in f:
        _, utt_id, _, s_time, offset, _, _, spk_id, _ = line.split("\t")
        s_time, offset = float(s_time), float(offset)
        if utt_id not in segments:
            segments[utt_id] = {}
            overlap_segs[utt_id] = []
            non_overlap_segs[utt_id] = []
            max_len[utt_id] = -1

        if spk_id not in segments[utt_id]:
            segments[utt_id][spk_id] = []

        segments[utt_id][spk_id].append(
            [int(round(s_time, 2) * 100), int(round(s_time + offset, 2) * 100)]
        )
        if round(s_time + offset, 2) * 100 > max_len[utt_id]:
            max_len[utt_id] = int(round(s_time + offset, 2) * 100)

time_segs = {}
time_segs_one = {}
for utt_id in segments:
    time_segs[utt_id] = {}
    time_segs_one[utt_id] = [0] * (max_len[utt_id] + 1)
    for spk_id in segments[utt_id]:
        time_segs[utt_id][spk_id] = [0] * (max_len[utt_id] + 1)
        # print((max_len[utt_id] + 1))
        # print(segments[utt_id][spk_id])
        for seg in segments[utt_id][spk_id]:
            for i in range(seg[0], seg[1]):
                time_segs[utt_id][spk_id][i] = 1
                time_segs_one[utt_id][i] += 1


input_rttm_path = "/workspace/junyi/codebase/joint-optimization/exp_fairseq/joint/baseline_8k/inf/res_rttm_"


low_DERs = 100
for thr in range(1, 10):
    out = subprocess.check_output(
        [
            "perl",
            "SCTK-2.4.12/src/md-eval/md-eval.pl",
            f"-c 0.0",
            "-s %s" % (input_rttm_path + str(thr / 10)),
            f"-r {rttm_path}",
        ]
    )
    out = out.decode("utf-8")
    DER, MS, FA, SC = (
        float(out.split("/")[0]),
        float(out.split("/")[1]),
        float(out.split("/")[2]),
        float(out.split("/")[3]),
    )
    print(
        "Eval for threshold %2.2f: DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"
        % (thr / 10, DER, MS, FA, SC)
    )

    if DER <= low_DERs:
        low_DERs = DER
        low_thr = thr
print(f"DER on all {low_DERs} with thr {low_thr}")
# thr = low_thr
# input_rttm = input_rttm_path + str(thr / 10)
# segments_in = {}
# max_len_in = {}
# overlap_segs_in = {}
# non_overlap_segs_in = {}
# with open(input_rttm) as f:
#     for line in f:
#         _, utt_id, _, s_time, offset, _, _, spk_id, _, _ = line.split()
#         s_time, offset = float(s_time), float(offset)
#         if utt_id not in segments_in:
#             segments_in[utt_id] = {}
#             overlap_segs_in[utt_id] = []
#             non_overlap_segs_in[utt_id] = []
#             max_len_in[utt_id] = -1

#         if spk_id not in segments_in[utt_id]:
#             segments_in[utt_id][spk_id] = []

#         segments_in[utt_id][spk_id].append([int(round(s_time, 2) * 100), int(round(s_time + offset, 2) * 100)])
#         if round(s_time + offset, 2) * 100 > max_len_in[utt_id]:
#             max_len_in[utt_id] = int(round(s_time + offset, 2) * 100)

# time_segs_in = {}
# time_segs_one_in = {}
# for utt_id in segments_in:
#     time_segs_in[utt_id] = {}
#     time_segs_one_in[utt_id] = [0] * (max_len_in[utt_id] + 1)
#     for spk_id in segments_in[utt_id]:
#         time_segs_in[utt_id][spk_id] = [0] * (max_len_in[utt_id] + 1)
#         # print((max_len[utt_id] + 1))
#         # print(segments[utt_id][spk_id])
#         for seg in segments_in[utt_id][spk_id]:
#             for i in range(seg[0], seg[1]):
#                 time_segs_in[utt_id][spk_id][i] = 1
#                 time_segs_one_in[utt_id][i] += 1

# SS = []
# SQ = []
# QQ = []
# qq_ratio = []
# total_qq = 0
# for utt_id in segments_in:
#     total_qq2 = 0
#     for spk_id in segments_in[utt_id]:
#         start_ss = -1
#         dur_ss = 0
#         start_sq = -1
#         dur_sq = 0
#         start_qq = -1
#         dur_qq = 0
#         for i in range(max_len_in[utt_id]):
#             if i <= max_len[utt_id] and time_segs_one[utt_id][i] > 1 and time_segs_in[utt_id][spk_id][i] == 1:
#                 if start_ss == -1:
#                     start_ss = i
#                 dur_ss += 1
#             else:
#                 if start_ss != -1:
#                     SS.append(f"SPEAKER\t{utt_id}\t1\t{start_ss / 100}\t{dur_ss / 100}\t<NA>\t<NA>\t{spk_id}\t<NA>")
#                     start_ss = -1
#                     dur_ss = 0

#             if i <= max_len[utt_id] and time_segs_one[utt_id][i] == 1 and time_segs_in[utt_id][spk_id][i] == 1:
#                 if start_sq == -1:
#                     start_sq = i
#                 dur_sq += 1
#             else:
#                 if start_sq != -1:
#                     SQ.append(f"SPEAKER\t{utt_id}\t1\t{start_sq / 100}\t{dur_sq / 100}\t<NA>\t<NA>\t{spk_id}\t<NA>")
#                     start_sq = -1
#                     dur_sq = 0

#             # if i <= max_len[utt_id] and time_segs_one[utt_id][i] >= 1 and time_segs_in[utt_id][spk_id][i] == 1:
#             #     if start_qq == -1:
#             #         start_qq = i
#             #     dur_qq += 1
#             # else:
#             #     if i > max_len[utt_id]:
#             #         if start_qq == -1:
#             #             start_qq = i
#             #         dur_qq += 1
#             #     else:
#             #         if start_qq != -1:
#             #             total_qq += dur_qq
#             #             total_qq2 += dur_qq
#             #             start_qq = -1
#             #             dur_qq = 0

#             if i <= max_len[utt_id] and time_segs_one[utt_id][i] == 0 and time_segs_in[utt_id][spk_id][i] == 1:
#                 if start_qq == -1:
#                     start_qq = i
#                 dur_qq += 1
#             else:
#                 if i > max_len[utt_id]:
#                     if start_qq == -1:
#                         start_qq = i
#                     dur_qq += 1
#                 else:
#                     if start_qq != -1:
#                         QQ.append(f"SPEAKER\t{utt_id}\t1\t{start_qq / 100}\t{dur_qq / 100}\t<NA>\t<NA>\t{spk_id}\t<NA>")
#                         total_qq += dur_qq
#                         total_qq2 += dur_qq
#                         start_qq = -1
#                         dur_qq = 0

#         if start_ss != -1:
#             SS.append(f"SPEAKER\t{utt_id}\t1\t{start_ss / 100}\t{dur_ss / 100}\t<NA>\t<NA>\t{spk_id}\t<NA>")
#             start_ss = -1
#             dur_ss = 0

#         if start_sq != -1:
#             SQ.append(f"SPEAKER\t{utt_id}\t1\t{start_sq / 100}\t{dur_sq / 100}\t<NA>\t<NA>\t{spk_id}\t<NA>")
#             start_sq = -1
#             dur_sq = 0

#         if start_qq != -1:
#             QQ.append(f"SPEAKER\t{utt_id}\t1\t{start_qq / 100}\t{dur_qq / 100}\t<NA>\t<NA>\t{spk_id}\t<NA>")
#             total_qq += dur_qq
#             total_qq2 += dur_qq
#             start_qq = -1
#             dur_qq = 0
#     qq_ratio.append(total_qq2 / max_len[utt_id])

# with open(input_rttm.replace('rttm', 'SS_rttm'), 'w') as f:
#     f.write('\n'.join(SS) + '\n')

# with open(input_rttm.replace('rttm', 'SQ_rttm'), 'w') as f:
#     f.write('\n'.join(SQ) + '\n')

# with open(input_rttm.replace('rttm', 'QQ_rttm'), 'w') as f:
#     f.write('\n'.join(QQ) + '\n')

# out = subprocess.check_output(['perl', 'SCTK-2.4.12/src/md-eval/md-eval.pl', f"-c 0.0", '-s %s'%(input_rttm.replace('rttm', 'SS_rttm')), f"-r {rttm_path.replace('.rttm', '_SS.rttm')}"])
# out = out.decode('utf-8')
# DER, MS, FA, SC = float(out.split('/')[0]), float(out.split('/')[1]), float(out.split('/')[2]), float(out.split('/')[3])
# print("Eval for SS threshold %2.2f: DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"%(thr / 10, DER, MS, FA, SC))

# out = subprocess.check_output(['perl', 'SCTK-2.4.12/src/md-eval/md-eval.pl', f"-c 0.0", '-s %s'%(input_rttm.replace('rttm', 'SQ_rttm')), f"-r {rttm_path.replace('.rttm', '_SQ.rttm')}"])
# out = out.decode('utf-8')
# DER, MS, FA, SC = float(out.split('/')[0]), float(out.split('/')[1]), float(out.split('/')[2]), float(out.split('/')[3])
# print("Eval for SQ&QS threshold %2.2f: DER %2.2f%%, MS %2.2f%%, FA %2.2f%%, SC %2.2f%%\n"%(thr / 10, DER, MS, FA, SC))

# # out = subprocess.check_output(['perl', 'SCTK-2.4.12/src/md-eval/md-eval.pl', f"-c 0.0", '-s %s'%(input_rttm.replace('rttm', 'QQ_rttm')), f"-r {rttm_path.replace('.rttm', '_QQ.rttm')}"])
# # out = out.decode('utf-8')
# # DER, MS, FA, SC = float(out.split('/')[0]), float(out.split('/')[1]), float(out.split('/')[2]), float(out.split('/')[3])
# print("Eval for QQ second: " + str(total_qq / len(segments.keys())))
# print("Eval for QQ second: " + str(sum(qq_ratio) / len(qq_ratio)))
