#!/bin/bash
set -ex

export BASE=s3://mstar-data/cc100.txt.xz/
export LANGS=(af.txt.xz am.txt.xz ar.txt.xz as.txt.xz az.txt.xz be.txt.xz bg.txt.xz bn.txt.xz bn_rom.txt.xz br.txt.xz bs.txt.xz ca.txt.xz cs.txt.xz cy.txt.xz da.txt.xz de.txt.xz el.txt.xz en.txt.xz eo.txt.xz es.txt.xz et.txt.xz eu.txt.xz fa.txt.xz ff.txt.xz fi.txt.xz fr.txt.xz fy.txt.xz ga.txt.xz gd.txt.xz gl.txt.xz gn.txt.xz gu.txt.xz ha.txt.xz he.txt.xz hi.txt.xz hi_rom.txt.xz hr.txt.xz ht.txt.xz hu.txt.xz hy.txt.xz id.txt.xz ig.txt.xz is.txt.xz it.txt.xz ja.txt.xz jv.txt.xz ka.txt.xz kk.txt.xz km.txt.xz kn.txt.xz ko.txt.xz ku.txt.xz ky.txt.xz la.txt.xz lg.txt.xz li.txt.xz ln.txt.xz lo.txt.xz lt.txt.xz lv.txt.xz mg.txt.xz mk.txt.xz ml.txt.xz mn.txt.xz mr.txt.xz ms.txt.xz my.txt.xz my_zaw.txt.xz ne.txt.xz nl.txt.xz no.txt.xz ns.txt.xz om.txt.xz or.txt.xz pa.txt.xz pl.txt.xz ps.txt.xz pt.txt.xz qu.txt.xz rm.txt.xz ro.txt.xz ru.txt.xz sa.txt.xz sc.txt.xz sd.txt.xz si.txt.xz sk.txt.xz sl.txt.xz so.txt.xz sq.txt.xz sr.txt.xz ss.txt.xz su.txt.xz sv.txt.xz sw.txt.xz ta.txt.xz ta_rom.txt.xz te.txt.xz te_rom.txt.xz th.txt.xz tl.txt.xz tn.txt.xz tr.txt.xz ug.txt.xz uk.txt.xz ur.txt.xz ur_rom.txt.xz uz.txt.xz vi.txt.xz wo.txt.xz xh.txt.xz yi.txt.xz yo.txt.xz zh-Hans.txt.xz zh-Hant.txt.xz zu.txt.xz)

aws s3 cp $BASE${LANGS[$AWS_BATCH_JOB_ARRAY_INDEX]} - | xz -d | python3 /usr/local/bin/count_tokens.py | tee ${LANGS[$AWS_BATCH_JOB_ARRAY_INDEX]}.count
aws s3 cp ${LANGS[$AWS_BATCH_JOB_ARRAY_INDEX]}.count s3://lausen-mstar-dev/batch/${AWS_BATCH_JOB_ID//:/\/}/${LANGS[$AWS_BATCH_JOB_ARRAY_INDEX]}.count

