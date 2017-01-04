
# coding: utf-8

# In[10]:

import os
from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree

import shutil
from glob import glob
from difflib import SequenceMatcher

# Embed all sub-elements into course and make into one big file and find diff on course 13 and 14
# mypath = "data/BerkeleyX-Stat2.1x-2013_Spring/vertical"

# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# courses = []
# onlyfiles.sort()
# for file in onlyfiles:
#     courses.append(xml.etree.ElementTree.parse(mypath + "/" + file))

# counter = 0    
    
# queue = []    
# for course in courses:
#     root = course.getroot()
#     for child in root:
#         if child.tag and "url_name" in child.attrib:
#             queue.append([child.tag, child.attrib["url_name"]])
            
#     course.write("diffOutputs/diff" + str(counter) + ".xml")
#     counter += 1            
            
# while len(queue) != 0:
#     elem = queue.pop(0)
    
#     mypath = "data/BerkeleyX-Stat2.1x-2013_Spring/" + elem[0]
#     tree = xml.etree.ElementTree.parse(mypath + "/" + elem[1] + ".xml")
#     root = tree.getroot()
    
#     for child in root:
#         if child.tag and "url_name" in child.attrib:
#             queue.append([child.tag, child.attrib["url_name"]])
    
#     tree.write("diffOutputs/diff" + str(counter) + ".xml")
#     counter += 1
    
# # Check what's been replaced -> what's new, what's deleted, what's same
# mypath = "data/BerkeleyX-Stat_2.1x-1T2014/vertical"

    
# DIFF OUTPUTS OTHER

def getDiffDict():

    # Embed all sub-elements into course and make into one big file and find diff on course 13 and 14
    stats13 = "data/BerkeleyX-Stat2.1x-2013_Spring/vertical"
    stats14 = "data/BerkeleyX-Stat_2.1x-1T2014/vertical"
    
    delfx14 = "data/DelftX-AE1110x-1T2014/vertical"
    delfx15 = "data/DelftX-AE1110x-2T2015/vertical"

    stats13_files = [f for f in listdir(stats13) if isfile(join(stats13, f))]
    stats14_files = [f for f in listdir(stats14) if isfile(join(stats14, f))]
    
    delfx14_files = [f for f in listdir(delfx14) if isfile(join(delfx14, f))]
    delfx15_files = [f for f in listdir(delfx15) if isfile(join(delfx15, f))]
    
    stats13_files += delfx14_files
    stats14_files += delfx15_files

    # Find only files in 13 and not in 14
    deleted = []
    for file in stats13_files:
        if file not in stats14_files and file != "course.xml":
            deleted.append(file)

    # Find only files in 14 and not in 13
    newverticals = []
    for file in stats14_files:
        if file not in stats13_files and file != "course.xml":
            newverticals.append(file)

    # Find the courses that were the same
    same = []
    for file in stats13_files:
        if file in stats14_files and file != "course.xml":
            same.append(file)

    print(len(stats13_files))
    print(len(stats14_files))

    # print(deleted)
    # print(newverticals)
    # print(same)
    
    # No change
    # Change
    # Deleted

    stats13_root = "data/BerkeleyX-Stat2.1x-2013_Spring/"
    stats14_root = "data/BerkeleyX-Stat_2.1x-1T2014/"
    
    delfx14_root = "data/DelftX-AE1110x-1T2014/"
    delfx15_root = "data/DelftX-AE1110x-2T2015/"
    
    permitted = {"problem"}

    changed = []
    stayed_same = []

    queue = []
    all_paths = []
    for file in same:
        path = delfx14_root + "vertical/" + file
        if os.path.exists(path):
            queue.append(path)
            all_paths.append(path)
        else:
            path = stats13_root + "vertical/" + file
            queue.append(path)
            all_paths.append(path)
    #     tree = xml.etree.ElementTree.parse(path)
    #     root = tree.getroot()

        while len(queue) != 0:
            path = queue.pop(0)
            
            root_p = stats13_root

            tree = xml.etree.ElementTree.parse(path)
            root = tree.getroot()
            
            if "Delf" in path:
                root_p = delfx14_root

            for child in root:
                if child.tag and child.tag in permitted and "url_name" in child.attrib:
                    queue.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")
                    all_paths.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")

        first_course_paths = all_paths[:]

        ### SECOND COURSE VERTICAL
        queue = []
        all_paths = []
        
        path = delfx15_root + "vertical/" + file    
        if os.path.exists(path):
            queue.append(path)
            all_paths.append(path)
        else:
            path = stats14_root + "vertical/" + file
            queue.append(path)
            all_paths.append(path)
    #     tree = xml.etree.ElementTree.parse(path)
    #     root = tree.getroot()

        while len(queue) != 0:
            path = queue.pop(0)
            
            root_p = stats14_root
    
            tree = xml.etree.ElementTree.parse(path)
            root = tree.getroot()
            
            if "Delf" in path:
                root_p = delfx15_root

            for child in root:
                if child.tag and child.tag in permitted and "url_name" in child.attrib:
                    queue.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")
                    all_paths.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")

        second_course_paths = all_paths[:]

        allFiles = first_course_paths

        with open("first_course.txt", 'wb') as outfile:
            allFiles.sort()

            for filename in allFiles:
                if filename == "first_course.txt":
                    # don't want to copy the output into the output
                    continue
                with open(filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, outfile)

        allFiles = second_course_paths

        with open("second_course.txt", 'wb') as outfile:
            allFiles.sort()

            for filename in allFiles:
                if filename == "second_course.txt":
                    # don't want to copy the output into the output
                    continue
                with open(filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, outfile)

        text1 = open("first_course.txt").read()
        text2 = open("second_course.txt").read()

        m = SequenceMatcher(None, text1, text2)

        similarity = m.ratio()

    #     print(file)
    #     print(similarity)

        if (similarity > 0.3):
            stayed_same.append(file)
        else:
            changed.append(file)

        os.remove("first_course.txt")
        os.remove("second_course.txt")

    outp = {}
    outp["deleted"] = deleted
    outp["new"] = newverticals
    outp["changed"] = changed
    outp["same"] = stayed_same  
        
    return outp


def getDiffDictVideo():

    # Embed all sub-elements into course and make into one big file and find diff on course 13 and 14
    stats13 = "data/BerkeleyX-Stat2.1x-2013_Spring/vertical"
    stats14 = "data/BerkeleyX-Stat_2.1x-1T2014/vertical"
    
    delfx14 = "data/DelftX-AE1110x-1T2014/vertical"
    delfx15 = "data/DelftX-AE1110x-2T2015/vertical"

    stats13_files = [f for f in listdir(stats13) if isfile(join(stats13, f))]
    stats14_files = [f for f in listdir(stats14) if isfile(join(stats14, f))]
    
    delfx14_files = [f for f in listdir(delfx14) if isfile(join(delfx14, f))]
    delfx15_files = [f for f in listdir(delfx15) if isfile(join(delfx15, f))]
    
    stats13_files += delfx14_files
    stats14_files += delfx15_files

    # Find only files in 13 and not in 14
    deleted = []
    for file in stats13_files:
        if file not in stats14_files and file != "course.xml":
            deleted.append(file)

    # Find only files in 14 and not in 13
    newverticals = []
    for file in stats14_files:
        if file not in stats13_files and file != "course.xml":
            newverticals.append(file)

    # Find the courses that were the same
    same = []
    for file in stats13_files:
        if file in stats14_files and file != "course.xml":
            same.append(file)

    print(len(stats13_files))
    print(len(stats14_files))

    # print(deleted)
    # print(newverticals)
    # print(same)
    
    # No change
    # Change
    # Deleted

    stats13_root = "data/BerkeleyX-Stat2.1x-2013_Spring/"
    stats14_root = "data/BerkeleyX-Stat_2.1x-1T2014/"
    
    delfx14_root = "data/DelftX-AE1110x-1T2014/"
    delfx15_root = "data/DelftX-AE1110x-2T2015/"
    
    permitted = {"video"}

    changed = []
    stayed_same = []

    queue = []
    all_paths = []
    for file in same:
        path = delfx14_root + "vertical/" + file
        if os.path.exists(path):
            queue.append(path)
            all_paths.append(path)
        else:
            path = stats13_root + "vertical/" + file
            queue.append(path)
            all_paths.append(path)
    #     tree = xml.etree.ElementTree.parse(path)
    #     root = tree.getroot()

        while len(queue) != 0:
            path = queue.pop(0)
            
            root_p = stats13_root

            tree = xml.etree.ElementTree.parse(path)
            root = tree.getroot()
            
            if "Delf" in path:
                root_p = delfx14_root

            for child in root:
                if child.tag and child.tag in permitted and "url_name" in child.attrib:
                    queue.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")
                    all_paths.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")

        first_course_paths = all_paths[:]

        ### SECOND COURSE VERTICAL
        queue = []
        all_paths = []
        
        path = delfx15_root + "vertical/" + file    
        if os.path.exists(path):
            queue.append(path)
            all_paths.append(path)
        else:
            path = stats14_root + "vertical/" + file
            queue.append(path)
            all_paths.append(path)
    #     tree = xml.etree.ElementTree.parse(path)
    #     root = tree.getroot()

        while len(queue) != 0:
            path = queue.pop(0)
            
            root_p = stats14_root
    
            tree = xml.etree.ElementTree.parse(path)
            root = tree.getroot()
            
            if "Delf" in path:
                root_p = delfx15_root

            for child in root:
                if child.tag and child.tag in permitted and "url_name" in child.attrib:
                    queue.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")
                    all_paths.append(root_p + child.tag + "/" + child.attrib["url_name"] + ".xml")

        second_course_paths = all_paths[:]

        allFiles = first_course_paths

        with open("first_course.txt", 'wb') as outfile:
            allFiles.sort()

            for filename in allFiles:
                if filename == "first_course.txt":
                    # don't want to copy the output into the output
                    continue
                with open(filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, outfile)

        allFiles = second_course_paths

        with open("second_course.txt", 'wb') as outfile:
            allFiles.sort()

            for filename in allFiles:
                if filename == "second_course.txt":
                    # don't want to copy the output into the output
                    continue
                with open(filename, 'rb') as readfile:
                    shutil.copyfileobj(readfile, outfile)

        text1 = open("first_course.txt").read()
        text2 = open("second_course.txt").read()

        m = SequenceMatcher(None, text1, text2)

        similarity = m.ratio()

    #     print(file)
    #     print(similarity)

        if (similarity > 0.3):
            stayed_same.append(file)
        else:
            changed.append(file)

        os.remove("first_course.txt")
        os.remove("second_course.txt")

    outp = {}
    outp["deleted"] = deleted
    outp["new"] = newverticals
    outp["changed"] = changed
    outp["same"] = stayed_same  
        
    return outp

# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# courses = []
# onlyfiles.sort()
# for file in onlyfiles:
#     courses.append(xml.etree.ElementTree.parse(mypath + "/" + file).getroot())

# counter = 0    
    
# queue = []    
# for course in courses:
#     for child in course:
#         if child.tag and "url_name" in child.attrib:
#             queue.append([child.tag, child.attrib["url_name"]])
            
#     tree.write("diffOutputsOther/diff" + str(counter) + ".xml")
#     counter += 1            
            
# while len(queue) != 0:
#     elem = queue.pop(0)
    
#     mypath = "data/BerkeleyX-Stat_2.1x-1T2014/" + elem[0]
#     tree = xml.etree.ElementTree.parse(mypath + "/" + elem[1] + ".xml")
#     root = tree.getroot()
    
#     for child in root:
#         if child.tag and "url_name" in child.attrib:
#             queue.append([child.tag, child.attrib["url_name"]])
    
#     tree.write("diffOutputsOther/diff" + str(counter) + ".xml")
#     counter += 1
    
#### OUTPUT TXT FILE
    
# import shutil
# from glob import glob

# allFiles = []

# with open("firstout.txt", 'wb') as outfile:
#     for filename in glob('diffOutputs/*.xml'):
#         allFiles.append(filename)
    
#     allFiles.sort()
    
#     for filename in allFiles:
#         if filename == "firstout.txt":
#             # don't want to copy the output into the output
#             continue
#         with open(filename, 'rb') as readfile:
#             shutil.copyfileobj(readfile, outfile)

# allFiles = []

# with open("secondout.txt", 'wb') as outfile:
#     for filename in glob('diffOutputsOther/*.xml'):
#         allFiles.append(filename)
    
#     allFiles.sort()
    
#     for filename in allFiles:
#         if filename == "secondout.txt":
#             # don't want to copy the output into the output
#             continue
#         with open(filename, 'rb') as readfile:
#             shutil.copyfileobj(readfile, outfile)
    
        
# print(courses)
# print(tree)


# In[11]:

def getDiffOnProbAndVideo():

    d = getDiffDict()
    d1 = getDiffDictVideo()

    for change in d1["changed"]:
        if change not in d["changed"]:
            d["changed"].append(change)
            if change in d["same"]:
                d["same"].remove(change)
    
    return d
            



# In[12]:

# getDiffOnProbAndVideo()


# In[ ]:



