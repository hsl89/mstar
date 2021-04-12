# Obtaining CC-100

http://data.statmt.org/cc-100/ hosts a recreation of the dataset used for
training XLM-R. Here we provide AWS Batch setup to parallelize the download of
CC-100 splits accross many small container instances due to severe IP-based rate
limiting of statmt.org to around 300-500 KB/s.

We have downloaded the corpus and it's available at
s3://mstar-data/cc100.txt.xz/. We also have the original index files used for
creation of the corpus at s3://mstar-data/cc100_index/.


## Setup for reproducing

This is a rough guide for reproducing the setup. It works. (Another option is to
explore AWS EMR cluster instead of AWS Batch as the code then may become easier
to reuse.)

### 0. Setting up AWS Batch cluster via `tools/cloudformation_template.yaml`

Set up AWS Batch environment etc.

```
aws --profile mstar cloudformation create-stack --stack-name lausen-mstar --template-body file:///home/ANT.AMAZON.COM/lausen/src/mstar/tools/cc100/cloudformation_template.yaml --capabilities CAPABILITY_NAMED_IAM
```

### 1. Updating the Docker

Currently AWS Batch does not support specifying the Docker image at runtime, but
will always use the image tag specified in the
`tools/cloudformation_template.yaml`. Thus, build the correct image here and
push it to the standardized tag.

```
aws --profile mstar ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 216212465934.dkr.ecr.us-east-1.amazonaws.com

docker build -t lausen-mstar .

docker tag lausen-mstar:latest 216212465934.dkr.ecr.us-east-1.amazonaws.com/lausen-mstar:latest

docker push 216212465934.dkr.ecr.us-east-1.amazonaws.com/lausen-mstar:latest
```


### 2. Downloading the splits

#### Option A: One machine per language

```
aws --profile mstar batch submit-job --job-queue lausen-mstar-fargate --job-name lausen-cc100 --job-definition lausen-mstar-fargate --array-properties size=116 --container-overrides '{"command": ["/bin/bash", "-c", "set -ex; export base=http://data.statmt.org/cc-100/; urls=(\"af.txt.xz\" \"am.txt.xz\" \"ar.txt.xz\" \"as.txt.xz\" \"az.txt.xz\" \"be.txt.xz\" \"bg.txt.xz\" \"bn.txt.xz\" \"bn_rom.txt.xz\" \"br.txt.xz\" \"bs.txt.xz\" \"ca.txt.xz\" \"cs.txt.xz\" \"cy.txt.xz\" \"da.txt.xz\" \"de.txt.xz\" \"el.txt.xz\" \"en.txt.xz\" \"eo.txt.xz\" \"es.txt.xz\" \"et.txt.xz\" \"eu.txt.xz\" \"fa.txt.xz\" \"ff.txt.xz\" \"fi.txt.xz\" \"fr.txt.xz\" \"fy.txt.xz\" \"ga.txt.xz\" \"gd.txt.xz\" \"gl.txt.xz\" \"gn.txt.xz\" \"gu.txt.xz\" \"ha.txt.xz\" \"he.txt.xz\" \"hi.txt.xz\" \"hi_rom.txt.xz\" \"hr.txt.xz\" \"ht.txt.xz\" \"hu.txt.xz\" \"hy.txt.xz\" \"id.txt.xz\" \"ig.txt.xz\" \"is.txt.xz\" \"it.txt.xz\" \"ja.txt.xz\" \"jv.txt.xz\" \"ka.txt.xz\" \"kk.txt.xz\" \"km.txt.xz\" \"kn.txt.xz\" \"ko.txt.xz\" \"ku.txt.xz\" \"ky.txt.xz\" \"la.txt.xz\" \"lg.txt.xz\" \"li.txt.xz\" \"ln.txt.xz\" \"lo.txt.xz\" \"lt.txt.xz\" \"lv.txt.xz\" \"mg.txt.xz\" \"mk.txt.xz\" \"ml.txt.xz\" \"mn.txt.xz\" \"mr.txt.xz\" \"ms.txt.xz\" \"my.txt.xz\" \"my_zaw.txt.xz\" \"ne.txt.xz\" \"nl.txt.xz\" \"no.txt.xz\" \"ns.txt.xz\" \"om.txt.xz\" \"or.txt.xz\" \"pa.txt.xz\" \"pl.txt.xz\" \"ps.txt.xz\" \"pt.txt.xz\" \"qu.txt.xz\" \"rm.txt.xz\" \"ro.txt.xz\" \"ru.txt.xz\" \"sa.txt.xz\" \"si.txt.xz\" \"sc.txt.xz\" \"sd.txt.xz\" \"sk.txt.xz\" \"sl.txt.xz\" \"so.txt.xz\" \"sq.txt.xz\" \"sr.txt.xz\" \"ss.txt.xz\" \"su.txt.xz\" \"sv.txt.xz\" \"sw.txt.xz\" \"ta.txt.xz\" \"ta_rom.txt.xz\" \"te.txt.xz\" \"te_rom.txt.xz\" \"th.txt.xz\" \"tl.txt.xz\" \"tn.txt.xz\" \"tr.txt.xz\" \"ug.txt.xz\" \"uk.txt.xz\" \"ur.txt.xz\" \"ur_rom.txt.xz\" \"uz.txt.xz\" \"vi.txt.xz\" \"wo.txt.xz\" \"xh.txt.xz\" \"yi.txt.xz\" \"yo.txt.xz\" \"zh-Hans.txt.xz\" \"zh-Hant.txt.xz\" \"zu.txt.xz\"); wget --progress=dot:giga -O - $base${urls[$AWS_BATCH_JOB_ARRAY_INDEX]} | aws s3 cp - s3://lausen-mstar-dev/batch/${AWS_BATCH_JOB_ID//:/\/}/"]}'
```

You can follow the job status at https://console.aws.amazon.com/batch/v2/home?region=us-east-1#jobs/

As statmt throttles each machine to around 500KB/s, this will take around 2 days
to finish.


#### Option B: Multiple machines for large languages

To speed up the download of large language splits such as English 82G, you can
further submit multiple jobs for the large languages. (You may not want to
submit all of them at the same time to avoid overloading the statmt server):

(Here the number of jobs is chosen to be a divisor of the file size.)


```
aws --profile mstar batch submit-job --job-queue lausen-mstar-fargate --job-name lausen-cc100-en --job-definition lausen-mstar-fargate --array-properties size=33 --container-overrides '{"command": ["/bin/bash", "-c", "set -ex; export LANG=en.txt.xz; export BASE=http://data.statmt.org/cc-100/; export CONTENT_LENGTH=$(curl -sI $BASE$LANG | grep Content-Length | tr -dc \"0-9\"); curl --fail -r $(($AWS_BATCH_JOB_ARRAY_INDEX*$CONTENT_LENGTH/33))-$((($AWS_BATCH_JOB_ARRAY_INDEX+1)*$CONTENT_LENGTH/33-1)) -o $LANG.$AWS_BATCH_JOB_ARRAY_INDEX $BASE$LANG; aws s3 cp $LANG.$AWS_BATCH_JOB_ARRAY_INDEX s3://lausen-mstar-dev/batch/${AWS_BATCH_JOB_ID//:/\/}"]}'

aws --profile mstar batch submit-job --job-queue lausen-mstar-fargate --job-name lausen-cc100-id --job-definition lausen-mstar-fargate --array-properties size=80 --container-overrides '{"command": ["/bin/bash", "-c", "set -ex; export LANG=id.txt.xz; export BASE=http://data.statmt.org/cc-100/; export CONTENT_LENGTH=$(curl -sI $BASE$LANG | grep Content-Length | tr -dc \"0-9\"); curl --fail -r $(($AWS_BATCH_JOB_ARRAY_INDEX*$CONTENT_LENGTH/80))-$((($AWS_BATCH_JOB_ARRAY_INDEX+1)*$CONTENT_LENGTH/80-1)) -o $LANG.$AWS_BATCH_JOB_ARRAY_INDEX $BASE$LANG; aws s3 cp $LANG.$AWS_BATCH_JOB_ARRAY_INDEX s3://lausen-mstar-dev/batch/${AWS_BATCH_JOB_ID//:/\/}"]}'

aws --profile mstar batch submit-job --job-queue lausen-mstar-fargate --job-name lausen-cc100-ru --job-definition lausen-mstar-fargate --array-properties size=62 --container-overrides '{"command": ["/bin/bash", "-c", "set -ex; export LANG=ru.txt.xz; export BASE=http://data.statmt.org/cc-100/; export CONTENT_LENGTH=$(curl -sI $BASE$LANG | grep Content-Length | tr -dc \"0-9\"); curl --fail -r $(($AWS_BATCH_JOB_ARRAY_INDEX*$CONTENT_LENGTH/62))-$((($AWS_BATCH_JOB_ARRAY_INDEX+1)*$CONTENT_LENGTH/62-1)) -o $LANG.$AWS_BATCH_JOB_ARRAY_INDEX $BASE$LANG; aws s3 cp $LANG.$AWS_BATCH_JOB_ARRAY_INDEX s3://lausen-mstar-dev/batch/${AWS_BATCH_JOB_ID//:/\/}"]}'

aws --profile mstar batch submit-job --job-queue lausen-mstar-fargate --job-name lausen-cc100-vi --job-definition lausen-mstar-fargate --array-properties size=32 --container-overrides '{"command": ["/bin/bash", "-c", "set -ex; export LANG=vi.txt.xz; export BASE=http://data.statmt.org/cc-100/; export CONTENT_LENGTH=$(curl -sI $BASE$LANG | grep Content-Length | tr -dc \"0-9\"); curl --fail -r $(($AWS_BATCH_JOB_ARRAY_INDEX*$CONTENT_LENGTH/32))-$((($AWS_BATCH_JOB_ARRAY_INDEX+1)*$CONTENT_LENGTH/32-1)) -o $LANG.$AWS_BATCH_JOB_ARRAY_INDEX $BASE$LANG; aws s3 cp $LANG.$AWS_BATCH_JOB_ARRAY_INDEX s3://lausen-mstar-dev/batch/${AWS_BATCH_JOB_ID//:/\/}"]}'

aws --profile mstar batch submit-job --job-queue lausen-mstar-fargate --job-name lausen-cc100-sv --job-definition lausen-mstar-fargate --array-properties size=86 --container-overrides '{"command": ["/bin/bash", "-c", "set -ex; export LANG=sv.txt.xz; export BASE=http://data.statmt.org/cc-100/; export CONTENT_LENGTH=$(curl -sI $BASE$LANG | grep Content-Length | tr -dc \"0-9\"); curl --fail -r $(($AWS_BATCH_JOB_ARRAY_INDEX*$CONTENT_LENGTH/86))-$((($AWS_BATCH_JOB_ARRAY_INDEX+1)*$CONTENT_LENGTH/86-1)) -o $LANG.$AWS_BATCH_JOB_ARRAY_INDEX $BASE$LANG; aws s3 cp $LANG.$AWS_BATCH_JOB_ARRAY_INDEX s3://lausen-mstar-dev/batch/${AWS_BATCH_JOB_ID//:/\/}"]}'

```


Once you obtain all files of a split, you download them and join them, for example:

```
cat 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 >> en.txt.xz
```

### 3. Token count

```
aws --profile mstar batch submit-job --job-queue lausen-mstar-c5 --job-name count-cc100 --job-definition lausen-mstar-c5 --array-properties size=116 --container-overrides '{"command": ["/usr/local/bin/count_job.sh"] }'
```

You can further speed-up this job by splitting the large language files into
pieces, listing those independently in count_job.sh and increasing the
array-size accordingly. (This may in fact bbe needed for en.txt.xz) as piping
the data to python3 can run out of memory if python isn't processing the data
fast enough.)
