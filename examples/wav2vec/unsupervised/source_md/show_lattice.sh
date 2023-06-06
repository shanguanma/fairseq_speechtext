#!/bin/bash

# refrence:kaldi/egs/babel/s5c/local/show_lattice.sh
#          kaldi/egs/wsj/s5/utils/show_lattice.sh

# example: local/show_lattice.sh --output test-one-lattice/dev_man --format svg --mode save --lm-scale 0.5 --acoustic-scale 0.1 nc12m-06nc12may_0101-00162-00459 test-one-lattice/dev_man/lat.1.gz data/lang/words.txt

 
format=svg # pdf svg
mode=save # display save
lm_scale=0.0
acoustic_scale=0.0
rm_eps=true
output=
#end of config

. utils/parse_options.sh

if [ $# != 3 ]; then
   echo "usage: $0 [--mode display|save] [--format pdf|svg] <utt-id> <lattice-ark> <word-list>"
   echo "e.g.:  $0 utt-0001 \"test/lat.*.gz\" tri1/graph/words.txt"
   exit 1;
fi

. ./path.sh

uttid=$1
lat=$2
words=$3

tmpdir=$(mktemp -d /tmp/kaldi.XXXX); # trap "rm -r $tmpdir" EXIT # cleanup

gunzip -c $lat | lattice-to-fst --lm-scale=$lm_scale --acoustic-scale=$acoustic_scale --rm-eps=$rm_eps ark:- "scp,p:echo $uttid $tmpdir/$uttid.fst|" || exit 1;
! [ -s $tmpdir/$uttid.fst ] && \
  echo "Failed to extract lattice for utterance $uttid (not present?)" && exit 1;
fstdraw --portrait=true --osymbols=$words $tmpdir/$uttid.fst | dot -T${format} > $tmpdir/$uttid.${format}

if [ "$(uname)" == "Darwin" ]; then
    doc_open=open
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    doc_open=xdg-open
elif [ $mode == "display" ] ; then
        echo "Can not automaticaly open file on your operating system"
        mode=save
fi

if [ ! -z $output ]; then
   
   cp $tmpdir/$uttid.${format} $output
   cp $tmpdir/$uttid.fst $output
fi

[ $mode == "display" ] && $doc_open $tmpdir/$uttid.${format}
[[ $mode == "display" && $? -ne 0 ]] && echo "Failed to open ${format} format." && mode=save
#[ $mode == "save" ] && echo "Saving to $uttid.${format}" && cp $tmpdir/$uttid.${format} .

[ $format == "pdf" ] && evince $tmpdir/$uttid.pdf
[ $format == "svg" ] && eog $tmpdir/$uttid.svg

# office form to delete tmpdir 
trap "rm -r $tmpdir" EXIT # cleanup
# normal form to delete tmpdir
# rm -rf $tmpdir
exit 0

