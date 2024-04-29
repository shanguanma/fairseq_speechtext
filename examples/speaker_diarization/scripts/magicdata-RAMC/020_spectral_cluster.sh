#!/usr/bin/env bash


stage=0
stop_stage=1000

. utils/parse_options.sh
#. path_for_nn_vad.sh
#. path_for_fsq_sptt.sh # hltsz
. path_for_fsq_speechtext.sh # sribd


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ];then
  echo "spectral clustering ...."
  python scripts/magicdata-RAMC/020_spectral_cluster.py
  # SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r data/magicdata-RAMC_debug/test/rttm_debug2 -s data/magicdata-RAMC/test/rttm_predict
  # result:35.04/30.22/1.70/3.12
  #  SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r data/magicdata-RAMC_debug/test/rttm -s data/magicdata-RAMC/test/rttm_predict
  # result:
  # WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#34.59/30.26/0.01/4.32

fi

#if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ];then
#   echo "cpu get embedding"
#	python scripts/magicdata-RAMC/020_spectral_cluster_v2.py
#   echo "finish"
#fi

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
#     echo "clustering ......."
#     python scripts/magicdata-RAMC/020_spectral_cluster2.py
# fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    echo "oracle clustering ......."
    python scripts/magicdata-RAMC/020_spectral_cluster2_ora.py
   #  SCTK-2.4.12/src/md-eval/md-eval.pl -c 0.25 -r data/magicdata-RAMC_debug/test/rttm -s data/magicdata-RAM
#C_debug/test/rttm_debug2
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000169 speaking more than once at time 124.835
#WARNING:  speaker G00000000 speaking more than once at time 242.056
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 359.176
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000168 speaking more than once at time 388.331
#WARNING:  speaker G00000000 speaking more than once at time 646.813
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000168 speaking more than once at time 906.248
#WARNING:  speaker G00000000 speaking more than once at time 993.704
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000168 speaking more than once at time 1238.088
#WARNING:  speaker G00000000 speaking more than once at time 1256.272
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000169 speaking more than once at time 1363.458
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000168 speaking more than once at time 1471.048
#WARNING:  speaker G00000000 speaking more than once at time 1602.984
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000169 speaking more than once at time 1780.168
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000000 speaking more than once at time 447.416
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000464 speaking more than once at time 452.378
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 528.962
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000131 speaking more than once at time 532.848
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000185 speaking more than once at time 1421.653
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000290 speaking more than once at time 1526.856
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 280.592
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 406.701
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000583 speaking more than once at time 971.588
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000584 speaking more than once at time 1181.130
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1333.227
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1438.255
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1455.758
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000583 speaking more than once at time 1714.851
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000639 speaking more than once at time 1024.776
#WARNING:  speaker G00000000 speaking more than once at time 1499.859
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 324.970
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000364 speaking more than once at time 1355.424
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 371.369
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000231 speaking more than once at time 933.760
#WARNING:  speaker G00000000 speaking more than once at time 1094.816
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000231 speaking more than once at time 1173.609
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000230 speaking more than once at time 1183.944
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1236.736
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1503.376
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1542.736
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1584.552
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1600.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1656.000
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000231 speaking more than once at time 1671.752
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1737.416
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000230 speaking more than once at time 1752.809
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000231 speaking more than once at time 1802.152
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 824.512
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000283 speaking more than once at time 1151.936
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000282 speaking more than once at time 1329.656
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1500.328
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1501.544
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1765.672
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1795.024
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1844.736
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000283 speaking more than once at time 1915.304
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 581.612
#WARNING:  speaker G00000000 speaking more than once at time 745.872
#WARNING:  speaker G00000000 speaking more than once at time 1265.952
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 490.930
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000671 speaking more than once at time 922.378
#WARNING:  speaker G00000000 speaking more than once at time 1101.386
#WARNING:  speaker G00000000 speaking more than once at time 1101.954
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1106.322
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1110.122
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1151.858
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000672 speaking more than once at time 1172.058
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000671 speaking more than once at time 1292.322
#WARNING:  speaker G00000000 speaking more than once at time 1512.938
#WARNING:  speaker G00000000 speaking more than once at time 1541.690
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1841.738
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000672 speaking more than once at time 1848.514
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000000 speaking more than once at time 51.944
#WARNING:  speaker G00000000 speaking more than once at time 168.248
#WARNING:  speaker G00000000 speaking more than once at time 929.808
#WARNING:  speaker G00000000 speaking more than once at time 959.712
#WARNING:  speaker G00000000 speaking more than once at time 1400.736
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 103.272
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 506.368
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000045 speaking more than once at time 735.968
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000046 speaking more than once at time 1088.608
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 799.053
#WARNING:  speaker G00000000 speaking more than once at time 1017.744
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000000 speaking more than once at time 53.256
#WARNING:  speaker G00000000 speaking more than once at time 268.505
#WARNING:  speaker G00000000 speaking more than once at time 407.201
#WARNING:  speaker G00000000 speaking more than once at time 433.528
#WARNING:  speaker G00000000 speaking more than once at time 757.544
#WARNING:  speaker G00000000 speaking more than once at time 871.482
#WARNING:  speaker G00000000 speaking more than once at time 1035.712
#WARNING:  speaker G00000000 speaking more than once at time 1057.248
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000092 speaking more than once at time 1142.634
#WARNING:  speaker G00000000 speaking more than once at time 1162.832
#WARNING:  speaker G00000000 speaking more than once at time 1164.312
#WARNING:  speaker G00000000 speaking more than once at time 1241.203
#WARNING:  speaker G00000000 speaking more than once at time 1260.024
#WARNING:  speaker G00000000 speaking more than once at time 1308.680
#WARNING:  speaker G00000000 speaking more than once at time 1588.259
#WARNING:  speaker G00000000 speaking more than once at time 1731.002
#WARNING:  speaker G00000000 speaking more than once at time 1732.232
#WARNING:  speaker G00000000 speaking more than once at time 1741.378
#WARNING:  speaker G00000000 speaking more than once at time 1746.944
#WARNING:  speaker G00000000 speaking more than once at time 1751.770
#WARNING:  speaker G00000000 speaking more than once at time 1791.312
#WARNING:  speaker G00000000 speaking more than once at time 1791.968
#WARNING:  speaker G00000000 speaking more than once at time 1825.248
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000243 speaking more than once at time 73.258
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000537 speaking more than once at time 297.482
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 357.992
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000284 speaking more than once at time 482.832
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000285 speaking more than once at time 1135.592
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000284 speaking more than once at time 1421.712
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000285 speaking more than once at time 1442.656
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1508.323
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1630.563
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000284 speaking more than once at time 1798.032
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000285 speaking more than once at time 1886.980
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000326 speaking more than once at time 387.722
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000327 speaking more than once at time 632.136
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000326 speaking more than once at time 839.436
#WARNING:  speaker G00000000 speaking more than once at time 1013.128
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1241.096
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000327 speaking more than once at time 1695.756
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000490 speaking more than once at time 405.167
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000489 speaking more than once at time 678.490
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1127.747
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000490 speaking more than once at time 1860.903
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000000 speaking more than once at time 24.746
#WARNING:  speaker G00000000 speaking more than once at time 1266.368
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000528 speaking more than once at time 1780.944
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000534 speaking more than once at time 116.488
#WARNING:  speaker G00000000 speaking more than once at time 361.608
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 543.008
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 666.832
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1035.624
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000533 speaking more than once at time 1793.704
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 377.309
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000621 speaking more than once at time 477.125
#WARNING:  speaker G00000000 speaking more than once at time 547.109
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 855.493
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000622 speaking more than once at time 864.773
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 899.869
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 916.613
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 992.837
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1235.525
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000621 speaking more than once at time 1722.885
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000622 speaking more than once at time 1774.429
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000621 speaking more than once at time 1818.725
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000680 speaking more than once at time 130.024
#WARNING:  speaker G00000000 speaking more than once at time 209.704
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000680 speaking more than once at time 256.099
#WARNING:  speaker G00000000 speaking more than once at time 280.673
#WARNING:  speaker G00000000 speaking more than once at time 312.328
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 804.904
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1129.232
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1223.816
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1411.528
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000680 speaking more than once at time 1536.809
#WARNING:  speaker G00000000 speaking more than once at time 1730.288
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000392 speaking more than once at time 795.224
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 809.608
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000391 speaking more than once at time 1701.352
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1307.916
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000430 speaking more than once at time 1667.180
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000429 speaking more than once at time 1823.436
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000662 speaking more than once at time 31.144
#WARNING:  speaker G00000000 speaking more than once at time 265.232
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 358.408
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 752.928
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000661 speaking more than once at time 900.488
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1328.008
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1504.744
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000662 speaking more than once at time 1852.488
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000000 speaking more than once at time 424.078
#WARNING:  speaker G00000000 speaking more than once at time 726.736
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000141 speaking more than once at time 1731.995
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000322 speaking more than once at time 1789.851
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000155 speaking more than once at time 1816.256
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 950.478
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000341 speaking more than once at time 1127.215
#WARNING:  speaker G00000000 speaking more than once at time 1579.866
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#WARNING:  speaker G00000000 speaking more than once at time 334.738
#WARNING:  speaker G00000000 speaking more than once at time 783.496
#WARNING:  speaker G00000000 speaking more than once at time 783.784
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000094 speaking more than once at time 1272.240
#WARNING:  speaker G00000000 speaking more than once at time 1351.680
#WARNING:  speaker G00000000 speaking more than once at time 1725.120
#1.85/1.85/0.00/0.00
fi
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
    echo "clustering ......."
    vad_threshold="0.6 0.7 0.8 0.9 0.91 0.92"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=3
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.5 0.6 0.7 0.8 0.9 0.91 0.92"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 4 > logs/020_spectral_cluster_stage4.log 2 >&1
   # result: cat logs/020_spectral_cluster_stage4.log
   # DER score
   # DER  MS FA SC
   #10.00/2.83/0.22/6.95
   #11.24/4.40/0.13/6.71
   #13.47/7.01/0.07/6.39
   #18.60/12.79/0.03/5.78
   #34.59/30.26/0.01/4.32
   #37.19/33.05/0.01/4.12
   #40.03/36.10/0.01/3.92

 fi

 if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   echo "CDER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.5 0.6 0.7 0.8 0.9 0.91 0.92"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 5 > logs/020_spectral_cluster_stage5 2>&1
   #  grep -r "Avg CDER" logs/020_spectral_cluster_stage5
   # Avg CDER : 0.218
   # Avg CDER : 0.218
   # Avg CDER : 0.224
   # Avg CDER : 0.218
   # Avg CDER : 0.234
   # Avg CDER : 0.237
   # Avg CDER : 0.249
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.5 0.6 0.7 0.8 0.9 0.91 0.92"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 6 > logs/020_spectral_cluster_stage6.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage6.log
   # DER score collar=0.0
   # 18.67/4.36/4.06/10.24
   # 19.36/6.82/2.75/9.79
   # 21.57/10.72/1.63/9.22
   # 27.07/18.17/0.82/8.08
   # 42.03/35.99/0.28/5.76
   # 44.42/38.71/0.25/5.46
   # 47.01/41.64/0.22/5.15
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ];then
    echo "clustering ......."
    vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=3
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 9 --stop-stage 9 > logs/020_spectral_cluster_stage9.log 2 >&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage9.log
#DER score
#11.09/0.18/3.03/7.89
#9.22/0.42/1.12/7.69
#8.94/0.95/0.57/7.42
#9.33/1.76/0.34/7.24
#9.63/2.24/0.27/7.12
#9.70/2.36/0.26/7.09
#9.74/2.45/0.25/7.05
#10.09/2.95/0.21/6.93
#10.18/3.08/0.20/6.91
#10.25/3.20/0.19/6.86
 fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 10  --stop-stage 10 > logs/020_spectral_cluster_stage10.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage10.log
   # Avg CDER : 0.234
#Avg CDER : 0.225
#Avg CDER : 0.221
#Avg CDER : 0.218
#Avg CDER : 0.219
#Avg CDER : 0.217
#Avg CDER : 0.217
#Avg CDER : 0.218
#Avg CDER : 0.217
#Avg CDER : 0.217
fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.1 0.2 0.3 0.4 0.45 0.46 0.47 0.51 0.52 0.53"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 11 --stop-stage 11 > logs/020_spectral_cluster_stage11.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage11.log
   # DER score collar=0.0
#24.44/0.41/12.33/11.70
#21.69/0.86/9.44/11.39
#19.71/1.76/6.95/10.99
#18.85/2.96/5.22/10.67
#18.70/3.60/4.61/10.49
#18.69/3.75/4.50/10.44
#18.67/3.88/4.39/10.39
#18.69/4.53/3.95/10.21
#18.72/4.70/3.85/10.17
#18.72/4.87/3.74/10.11
fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=3
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 13 --stop-stage 13 > logs/020_spectral_cluster_stage13.log 2 >&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage13.log
   # DER score
#9.22/0.42/1.12/7.69
#9.07/0.48/0.96/7.62
#9.00/0.56/0.84/7.60
#8.97/0.61/0.79/7.57
#8.92/0.72/0.69/7.51
#8.92/0.81/0.64/7.47
#8.93/0.87/0.60/7.46
#8.94/0.95/0.57/7.42  #for vad_threshold==0.30
#8.98/1.03/0.54/7.41
#9.01/1.10/0.52/7.40
#9.09/1.23/0.47/7.39
#9.15/1.38/0.42/7.34
#9.24/1.60/0.37/7.27
fi

if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 14  --stop-stage 14 > logs/020_spectral_cluster_stage14.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage14.log
   # Avg CDER : 0.225
#Avg CDER : 0.223
#Avg CDER : 0.222
#Avg CDER : 0.221
#Avg CDER : 0.220
#Avg CDER : 0.220
#Avg CDER : 0.220
#Avg CDER : 0.221
#Avg CDER : 0.220
#Avg CDER : 0.220
#Avg CDER : 0.221
#Avg CDER : 0.219
#Avg CDER : 0.219
fi

if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=3
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 15 --stop-stage 15 > logs/020_spectral_cluster_stage15.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage15.log
   # DER score collar=0.0
#21.69/0.86/9.44/11.39
#21.31/0.99/9.02/11.30
#21.02/1.12/8.65/11.25
#20.86/1.21/8.44/11.22
#20.35/1.40/7.84/11.12
#20.10/1.54/7.50/11.06
#19.89/1.64/7.20/11.05
#19.71/1.76/6.95/10.99
#19.58/1.89/6.72/10.97
#19.46/1.99/6.52/10.95
#19.26/2.20/6.16/10.90
#19.09/2.44/5.83/10.83
#18.95/2.73/5.50/10.72

fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
    echo "clustering ......."
    vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    #vad_threshold="0.5"
    vad_type="transformer_vad"
    chunk_size=2
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=2
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 18 --stop-stage 18 > logs/020_spectral_cluster_stage18.log 2 >&1
   # grep -v 'WARNING' logs/020_spectral_cluster_stage18.log
   # DER score
#8.06/0.70/1.05/6.31
#8.00/0.79/0.90/6.31
#7.96/0.87/0.79/6.30
#7.97/0.94/0.74/6.29
#8.01/1.11/0.65/6.24
#8.06/1.22/0.61/6.24
#8.11/1.32/0.57/6.23
#8.20/1.42/0.54/6.24
#8.26/1.51/0.51/6.23
#8.32/1.61/0.49/6.23
#8.45/1.79/0.44/6.22
#8.57/2.01/0.39/6.17
#8.71/2.26/0.35/6.10
fi

if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=2
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 19  --stop-stage 19 > logs/020_spectral_cluster_stage19.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage19.log
   # Avg CDER : 0.233
#Avg CDER : 0.231
#Avg CDER : 0.229
#Avg CDER : 0.229
#Avg CDER : 0.227
#Avg CDER : 0.228
#Avg CDER : 0.228
#Avg CDER : 0.229
#Avg CDER : 0.227
#Avg CDER : 0.226
#Avg CDER : 0.225
#Avg CDER : 0.224
#Avg CDER : 0.223
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=2
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 20 --stop-stage 20 > logs/020_spectral_cluster_stage20.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage20.log
#   DER score collar=0.0
#20.43/1.40/9.06/9.97
#20.14/1.57/8.63/9.94
#19.90/1.69/8.29/9.92
#19.79/1.81/8.09/9.89
#19.38/2.07/7.50/9.81
#19.21/2.25/7.18/9.79
#19.04/2.41/6.88/9.76
#18.94/2.55/6.63/9.76
#18.84/2.69/6.42/9.73
#18.76/2.84/6.21/9.72
#18.63/3.11/5.85/9.68
#18.55/3.43/5.51/9.61
#18.46/3.76/5.20/9.51
fi


if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ];then
    echo "clustering ......."
    #vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
    vad_threshold="0.31 0.32 0.34 0.36 0.38"
    vad_type="transformer_vad"
    chunk_size=1.5
    skip_chunk_size=0.93
    test_set_dir="data/magicdata-RAMC/test/"
    for name in $vad_threshold;do
     python scripts/magicdata-RAMC/020_spectral_cluster_common.py\
         --vad_threshold $name\
         --predict_vad_path_dir $test_set_dir/predict_vad_${name} \
         --test_set_dir $test_set_dir\
         --predict_rttm_path $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}\
         --chunk_size $chunk_size\
         --skip_chunk_size $skip_chunk_size\
         --vad_type $vad_type


    done
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ];then
   echo "DER score"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.25 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 22 --stop-stage 22 > logs/020_spectral_cluster_stage22.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage22.log
fi

if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ];then
   echo "CDER score"
   #note:collar of CDER score default set is equal to zero.
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   skip_chunk_size=0.93
   . path_for_nn_vad.sh
   for name in $vad_threshold;do
     python clustering_based/cder/score.py  -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   # bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 23  --stop-stage 23 > logs/020_spectral_cluster_stage23.log 2>&1
   #  grep -r 'Avg CDER' logs/020_spectral_cluster_stage23.log
fi

if [ ${stage} -le 24 ] && [ ${stop_stage} -ge 24 ];then
   echo "DER score collar=0.0"
   sctk_dir=SCTK-2.4.12/src/md-eval
   vad_threshold="0.2 0.22 0.24 0.25 0.27 0.28 0.29 0.30 0.31 0.32 0.34 0.36 0.38"
   test_set_dir="data/magicdata-RAMC/test/"
   vad_type="transformer_vad"
   chunk_size=1.5
   skip_chunk_size=0.93
   for name in $vad_threshold;do
     $sctk_dir/md-eval.pl -c 0.0 -r $test_set_dir/rttm -s $test_set_dir/rttm_predict_${name}_${vad_type}_chunk_size_${chunk_size}_skip_chunk_size_${skip_chunk_size}
   done
   #  #  bash scripts/magicdata-RAMC/020_spectral_cluster.sh --stage 24 --stop-stage 24 > logs/020_spectral_cluster_stage24.log 2>&1
   #  grep -v 'WARNING' logs/020_spectral_cluster_stage24.log
fi
