pids=() # initialize pids
for gid in 0 1 2 3 ; do
(
    export CUDA_VISIBLE_DEVICES=${gid}
    bash scripts/extract_speaker_embed_libricss.sh ${gid}

) &
pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
