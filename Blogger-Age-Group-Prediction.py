import os.path
from  collections import defaultdict
import string
import subprocess
import time

BASEDIR= ".\\"

ANALYSISDIR='analysis' 
WORKINGDIR ='working'   

DEVDIR  = 'data-devtest'
TRAINDIR = 'data-train'
TESTDIR  = 'data-held-out-test'


m = defaultdict(int)
count_dict= defaultdict(int)
d_words = defaultdict(int)
t_count_dict= defaultdict(int)



def load_freq_words():
    d_words.clear()
    datafile=os.path.join('./most-freq-words.txt')
    with open(datafile,'r')as df:
        for line in df:
            line = line.split()
            d_words[line[0]] = len(d_words.keys())+1 
    df.close
    return d_words


def process_dirs(process_dir):
    af =  open(os.path.join(WORKINGDIR,'svm'+process_dir+'.dat'),'w')

    for f in os.listdir(process_dir):
        if not os.path.isdir(process_dir+'/'+f):
           svmRow=do_count(f,process_dir )
           af.write(svmRow+'\n')
        else:
            for g in os.listdir(process_dir+'/'+f):
                if not os.path.isdir(process_dir +'/'+f+'/'+g):
                    svmRow=do_count(g,process_dir +'/'+f)
                    af.write(svmRow+'\n')
                else: 
                    for h in os.listdir(process_dir+'/'+f+'/'+g):
                        svmRow=do_count(h,process_dir +'/'+f+'/'+g)
                        af.write(svmRow+'\n')
    af.close()
    return None


def do_count(blog_file, blog_dir):
    global d_words
    global count_dict
    global m   

    
    count_dict.clear()
    m.clear()
    
    dfile=os.path.join(blog_dir,blog_file)
    
    with open(dfile,'r')as df:
        data_contents=df.read()
        xmlPosts = ''
        iEnd=0
        i = 0
        svmRow=''

        while iEnd <> -1:
            iStart = data_contents.find('<post>', i)
            if iStart == -1:
                break
            iEnd = data_contents.find('</post>', i)
            xmlPost = data_contents[iStart+6:iEnd-1]
            xmlPosts = xmlPosts + ' ' + xmlPost
            i = iEnd+7

        data_list = xmlPosts.split()
        
        for k in data_list:
            count_dict[k] += 1
            t_count_dict[k] += 1
            
        for k in count_dict.keys():
            if d_words.has_key(k):
                m[d_words[k]]= count_dict[k]
         

        for k in sorted(m.keys()):
            svmRow = svmRow + str(k) + ':' + str(m[k]) + ' '

        if blog_dir.find('10s') <> -1:
            svmTarget ='+1'
        else:
            svmTarget ='-1'
            
        svmRow = svmTarget + ' ' + svmRow  + " # " + blog_file
    return svmRow

def printNwrite(output, data_dir):
    global t1
    t2=time.clock()
    x = 'Elapsed Time from Start (secs) = '+ str(t2-t1)
    output = output + x
    print output
    print ''
    
    text_file = open(BASEDIR+'\\'+ANALYSISDIR+'\\svm_'+data_dir+'.txt',"w")
    text_file.write (output)
    text_file.close()
    
    return None

def writeCSV_total_wordcount():
    global t_count_dict

    text_file =open(ANALYSISDIR+'/'+'word_count.csv','wb')
    
    for k in sorted (t_count_dict, key=t_count_dict.get,reverse=True):
         text_file.write (k+','+ str(t_count_dict[k])+ '\r\n')

    text_file.close()
    
    
def printNwrite(output, data_dir):
    global t1
    t2=time.clock()
    x = 'Elapsed Time from Start (secs) = '+ str(t2-t1)
    output = output + x
    print output
    print ''
    
    text_file = open(BASEDIR+'\\'+ANALYSISDIR+'\\svm_'+data_dir+'.txt',"w")
    text_file.write (output)
    text_file.close()
    
    return None

if __name__ =='__main__':
    
    t1=time.clock()

    d_words = load_freq_words()
    
    process_dirs (DEVDIR)
    process_dirs (TRAINDIR)
    process_dirs (TESTDIR)

    # Create the 2 SVM models  from dev-test data and  train data sets

 #   svm_learn_args = ' -x 1 ' + WORKINGDIR+'/svm'+DEVDIR+'.dat '+ ANALYSISDIR + '/model-'+ DEVDIR
    svm_learn_args = WORKINGDIR+'/svm'+DEVDIR+'.dat '+ ANALYSISDIR + '/model-'+ DEVDIR
 
    output = subprocess.check_output(BASEDIR + 'svm_learn.exe ' + svm_learn_args)
    printNwrite(output, 'learn-'+DEVDIR)

    svm_learn_args = ' -x 1 ' + WORKINGDIR+'/svm'+TRAINDIR+'.dat '+ ANALYSISDIR + '/model-'+ TRAINDIR
    output = subprocess.check_output(BASEDIR + 'svm_learn.exe ' + svm_learn_args)
    printNwrite(output, 'learn-'+TRAINDIR)

    # Apply the TRAIN generated model against the HELD-OUT data set
    
    svm_classify_args = ' ' + WORKINGDIR+'/svm'+TESTDIR+'.dat '+ ANALYSISDIR + '/model-'+ DEVDIR   + ' ' + ANALYSISDIR +'/predictions-dev-model-'+TESTDIR +'.txt'
    output = subprocess.check_output(BASEDIR + 'svm_classify.exe ' + svm_classify_args)
    printNwrite(output, 'classify-dev-model-'+TESTDIR)

    svm_classify_args = ' ' + WORKINGDIR+'/svm'+TESTDIR+'.dat '+ ANALYSISDIR + '/model-'+ TRAINDIR + ' ' + ANALYSISDIR +'/predictions-train-model-'+TESTDIR+'.txt'
    output = subprocess.check_output(BASEDIR + 'svm_classify.exe ' + svm_classify_args)
    printNwrite(output, 'classify-train-model-'+TESTDIR)


   # Apply the generated model against the  alternate data set
 
  
    svm_classify_args = WORKINGDIR+'/svm'+TRAINDIR+'.dat '+ ANALYSISDIR + '/model-'+ DEVDIR + ' '+  ANALYSISDIR +'/predictions-dev-model-'+TRAINDIR+'.txt'
    output = subprocess.check_output(BASEDIR + 'svm_classify.exe ' + svm_classify_args)
    printNwrite(output, 'classify-dev-model-'+TRAINDIR)

    svm_classify_args = WORKINGDIR+'/svm'+DEVDIR+'.dat '+ ANALYSISDIR + '/model-'+ TRAINDIR + ' '+ ANALYSISDIR +'/predictions-dev-model-'+DEVDIR+'.txt'
    output = subprocess.check_output(BASEDIR + 'svm_classify.exe ' + svm_classify_args)
    printNwrite(output, 'classify-train-model-'+DEVDIR)

   # For discovery and anlaysis, apply the TRAIN model to its own data set.

    svm_classify_args = WORKINGDIR+'/svm'+TRAINDIR+'.dat '+ ANALYSISDIR + '/model-'+ TRAINDIR +' '+ ANALYSISDIR +'/predictions-train-model-'+TRAINDIR+'.txt'
    output = subprocess.check_output(BASEDIR + 'svm_classify.exe ' + svm_classify_args)
    printNwrite(output, 'classify-train-model-'+TRAINDIR)

   # Try the polynomial learning option in TRAIN and apply against HELD-OUT data set.

    svm_learn_args = ' -t 1 -d 3 ' + WORKINGDIR+'/svm'+TRAINDIR+'.dat '+ ANALYSISDIR + '/model-polyn-'+ TRAINDIR
    output = subprocess.check_output(BASEDIR + 'svm_learn.exe ' + svm_learn_args)
    printNwrite(output, 'learn-polyn-'+TRAINDIR)

    svm_classify_args = WORKINGDIR+'/svm'+TESTDIR+'.dat '+ ANALYSISDIR + '/model-polyn-'+ TRAINDIR + ' '+ ANALYSISDIR +'/predictions-train-polyn-model-'+TESTDIR+'.txt'
    output = subprocess.check_output(BASEDIR + 'svm_classify.exe ' + svm_classify_args)
    printNwrite(output, 'classify-polyn-model-'+TESTDIR)

  
    # NOT REQUIRED but WORKS - Write total word counts into CSV file
    # writeCSV_total_wordcount()

    
