import pandas as pd
import numpy as np
import collections
from collections import Counter

def coursesEveryUser(enr):
    b = dict(collections.Counter(enr.username).items())
    username = list(b.keys())
    num_course = list(b.values())
    df = pd.DataFrame({'username':username,'num_course':num_course})
    df.to_csv("coursesEveryUser.csv")
    

def usersEveryCourse(enr):
    b = dict(collections.Counter(enr.course_id).items())
    course_id = list(b.keys())
    num_user = list(b.values())
    df = pd.DataFrame({'course_id':course_id,'num_user':num_user})
    df.to_csv("usersEveryCourse.csv")


def coursesEveryEnrollment(enr):
    coursesEveryUser = pd.read_csv("coursesEveryUser.csv")
    df = enr
    df = df.merge(coursesEveryUser, on = 'username', how='left')
    df.to_csv("coursesEveryEnrollment_test.csv")

    
    
def numOfLog(log, enr):
    df = pd.DataFrame(np.zeros((log.shape[0],2)),columns = ['enrollment_id','num_log'])
    df["enrollment_id"] = enr["enrollment_id"]
    for i in range(enr.shape[0]):
        df["num_log"][i] = log[log.enrollment_id == i].shape[0]
    df.to_csv("numOfLog.csv")


if __name__ == '__main__':
    log = pd.read_csv("log_train.csv")
    enr_train = pd.read_csv("enrollment_train.csv")
    enr_test = pd.read_csv("enrollment_test.csv")
    coursesEveryUser(enr_train)
    usersEveryCourse(enr_train)
    coursesEveryEnrollment(enr_test)
    numOfLog(log, enr_train)
    log1 = log[log.enrollment_id == 1]
    nameList = []
    print(log1.to_string())
    print(log[(log.event == 'nagivate') & (log.source == 'server')])
    print(log.query("event == 'nagivate' and source == 'server'"))