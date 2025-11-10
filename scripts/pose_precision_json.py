# convert JSON files into proper precision, instead of very long dicimals. a json file corresponding to each video, 
# the folder contains a number of pose tracking for a numbe rof video files.
# run this convert before postProcessing.py and after fencingVideo_inference3.py
#(alphapose) yin@Dell-XPS:~/gitSources/AlphaPose$ python scripts/pose_precision_json.py --in results/Test_AVI
# importing libraries
import argparse
import os
import json


def convert_a_file(in_file,out_file):
    rawText = ""
    with open(in_file,"r") as f:
        rawText = f.readline()

    strings = rawText.split(',')

    out_strings = []
    out_strings.append(strings[0])
    #print("i=0    out_string=",out_strings)
    for i in range(1, len(strings)):

        word = strings[i].replace(' ','')    

        if word.replace('.','').isdigit() == True:
            ss = word.split('.')
            if len(ss) == 2: # a float number
                if len(ss[0]) == 1: # a number smaller than 10.0 
                    word_new = ss[0] + "." + ss[1][:3]
                else:
                    word_new = ss[0] + "." + ss[1][:1]
            else:
                word_new =  word  # a integer or something else
        else:   
            #if i > 50:
                #print("i=",i,"   word=",word, word[:-1].replace('.',''))
            if word[-1] == ']' and word[:-1].replace('.','').isdigit():
                #print("--------------------------")
                ss = word[:-1].split('.')
                if len(ss) == 2:
                    if len(ss[0]) == 1:
                        word_new = ss[0] + "." + ss[1][:3] + ']'
                    else:
                        word_new = ss[0] + "." + ss[1][:1] + ']'
                else:
                    word_new =  word
            else:
                #print("++++++++++++--------------------------")
                strings2 = word.split('[')
                if len(strings2) == 2:
                    if strings2[1].replace('.','').replace(' ','').isdigit():
                        ss = strings2[1].split('.')
                        if len(ss) == 2:
                            if len(ss[0]) > 1:
                                out_string3 = ss[0] + "." + ss[1][:1]
                            else:
                                out_string3 = ss[0] + "." + ss[1][:3]
                            word_new =  strings2[0] + '['+ out_string3
                        else:
                            word_new =  word
                    else:
                        word_new =  word
                else:
                    strings2 = word.split(':')
                    if len(strings2) == 2:
                        if strings2[1].replace('.','').replace(' ','').isdigit():
                            ss = strings2[1].split('.')
                            if len(ss) == 2:
                                if len(ss[0]) > 1:
                                    out_string3 = ss[0] + "." + ss[1][:1]
                                else:
                                    out_string3 = ss[0] + "." + ss[1][:3]
                                word_new =  strings2[0] + ':'+ out_string3
                            else:
                                word_new =  word
                        else:
                            word_new =  word
                    else:
                        word_new =  word


        #out_string = out_string + "," + out_string2
        out_strings.append(","+word_new)
        
    out_string = ''.join(out_strings)
    with open(out_file,"w") as f:
        f.write(out_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pose precision')
    parser.add_argument('--inDir', type=str, required=False,
                        help='input json file')
    parser.add_argument('--inFile', type=str, required=False,
                        help='input json file')

    args = parser.parse_args()


    alphaPose_resuslt_path = args.inDir

    input_results = os.listdir(alphaPose_resuslt_path)

    print(f'Folder {alphaPose_resuslt_path} contains {len(input_results)} json file to be processed. These files are:')
    print(input_results)

    for path in input_results:
        result = alphaPose_resuslt_path + '/'+ path
        in_file = result + "/alphapose-results.json"
        out_file = result + "/precision_results.json"

        print("")
        print("converting pose json file:", in_file)

        convert_a_file(in_file,out_file)

        # rawText = ""
        # with open(in_file,"r") as f:
        #     rawText = f.readline()

        # strings = rawText.split(',')

        # #print(f' items in string list is {len(strings)}', strings[:50])

        # out_strings = []
        # out_strings.append(strings[0])
        # #print("i=0    out_string=",out_strings)
        # for i in range(1, len(strings)):

        #     word = strings[i].replace(' ','')    

        #     if word.replace('.','').isdigit() == True:
        #         ss = word.split('.')
        #         if len(ss) == 2: # a float number
        #             if len(ss[0]) == 1: # a number smaller than 10.0 
        #                 word_new = ss[0] + "." + ss[1][:3]
        #             else:
        #                 word_new = ss[0] + "." + ss[1][:1]
        #         else:
        #             word_new =  word  # a integer or something else
        #     else:   
        #         #if i > 50:
        #             #print("i=",i,"   word=",word, word[:-1].replace('.',''))
        #         if word[-1] == ']' and word[:-1].replace('.','').isdigit():
        #             #print("--------------------------")
        #             ss = word[:-1].split('.')
        #             if len(ss) == 2:
        #                 if len(ss[0]) == 1:
        #                     word_new = ss[0] + "." + ss[1][:3] + ']'
        #                 else:
        #                     word_new = ss[0] + "." + ss[1][:1] + ']'
        #             else:
        #                 word_new =  word
        #         else:
        #             #print("++++++++++++--------------------------")
        #             strings2 = word.split('[')
        #             if len(strings2) == 2:
        #                 if strings2[1].replace('.','').replace(' ','').isdigit():
        #                     ss = strings2[1].split('.')
        #                     if len(ss) == 2:
        #                         if len(ss[0]) > 1:
        #                             out_string3 = ss[0] + "." + ss[1][:1]
        #                         else:
        #                             out_string3 = ss[0] + "." + ss[1][:3]
        #                         word_new =  strings2[0] + '['+ out_string3
        #                     else:
        #                         word_new =  word
        #                 else:
        #                     word_new =  word
        #             else:
        #                 strings2 = word.split(':')
        #                 if len(strings2) == 2:
        #                     if strings2[1].replace('.','').replace(' ','').isdigit():
        #                         ss = strings2[1].split('.')
        #                         if len(ss) == 2:    # rawText = ""
        # with open(in_file,"r") as f:
        #     rawText = f.readline()

        # strings = rawText.split(',')

        # #print(f' items in string list is {len(strings)}', strings[:50])

        # out_strings = []
        # out_strings.append(strings[0])
        # #print("i=0    out_string=",out_strings)
        # for i in range(1, len(strings)):

        #     word = strings[i].replace(' ','')    

        #     if word.replace('.','').isdigit() == True:
        #         ss = word.split('.')
        #         if len(ss) == 2: # a float number
        #             if len(ss[0]) == 1: # a number smaller than 10.0 
        #                 word_new = ss[0] + "." + ss[1][:3]
        #             else:
        #                 word_new = ss[0] + "." + ss[1][:1]
        #         else:
        #             word_new =  word  # a integer or something else
        #     else:   
        #         #if i > 50:
        #             #print("i=",i,"   word=",word, word[:-1].replace('.',''))
        #         if word[-1] == ']' and word[:-1].replace('.','').isdigit():
        #             #print("--------------------------")
        #             ss = word[:-1].split('.')
        #             if len(ss) == 2:
        #                 if len(ss[0]) == 1:
        #                     word_new = ss[0] + "." + ss[1][:3] + ']'
        #                 else:
        #                     word_new = ss[0] + "." + ss[1][:1] + ']'
        #             else:
        #                 word_new =  word
        #         else:
        #             #print("++++++++++++--------------------------")
        #             strings2 = word.split('[')
        #             if len(strings2) == 2:
        #                 if strings2[1].replace('.','').replace(' ','').isdigit():
        #                     ss = strings2[1].split('.')
        #                     if len(ss) == 2:
        #                         if len(ss[0]) > 1:
        #                             out_string3 = ss[0] + "." + ss[1][:1]
        #                         else:
        #                             out_string3 = ss[0] + "." + ss[1][:3]
        #                         word_new =  strings2[0] + '['+ out_string3
        #                     else:
        #                         word_new =  word
        #                 else:
        #                     word_new =  word
        #             else:
        #                 strings2 = word.split(':')
        #                 if len(strings2) == 2:
        #                     if strings2[1].replace('.','').replace(' ','').isdigit():
        #                         ss = strings2[1].split('.')
        #                         if len(ss) == 2:
        #                             if len(ss[0]) > 1:
        #                                 out_string3 = ss[0] + "." + ss[1][:1]
        #                             else:
        #                                 out_string3 = ss[0] + "." + ss[1][:3]
        #                             word_new =  strings2[0] + ':'+ out_string3
        #                         else:
        #                             word_new =  word
        #                     else:
        #                         word_new =  word
        #                 else:
        #                     word_new =  word


        #     #out_string = out_string + "," + out_string2
        #     out_strings.append(","+word_new)
            
        # out_string = ''.join(out_strings)
        # with open(out_file,"w") as f:
        #     f.write(out_string)
        #                             if len(ss[0]) > 1:
        #                                 out_string3 = ss[0] + "." + ss[1][:1]
        #                             else:
        #                                 out_string3 = ss[0] + "." + ss[1][:3]
        #                             word_new =  strings2[0] + ':'+ out_string3
        #                         else:
        #                             word_new =  word
        #                     else:
        #                         word_new =  word
        #                 else:
        #                     word_new =  word


        #     #out_string = out_string + "," + out_string2
        #     out_strings.append(","+word_new)
            
        print("converted file is written to", out_file)




