
import sys
import pickle

from datetime import datetime
import gzip


def convert_to_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4] + '.' + dxStr[4:]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3] + '.' + dxStr[3:]
        else:
            return dxStr


def convert_to_3digit_icd9(dxStr):
    if dxStr.startswith('E'):
        if len(dxStr) > 4:
            return dxStr[:4]
        else:
            return dxStr
    else:
        if len(dxStr) > 3:
            return dxStr[:3]
        else:
            return dxStr


if __name__ == '__main__':
    admissionFile = sys.argv[1]
    diagnosisFile = sys.argv[2]
    patientsFile = sys.argv[3]
    outFile = sys.argv[4]


    print('Collecting mortality information')
    pidDodMap = {}
    infd = gzip.open(patientsFile, 'rt', encoding='utf-8')
    header=infd.readline()
    print(header)
    print('patients header:', header)
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        dod_hosp = tokens[5]
        if len(dod_hosp) > 0:
            pidDodMap[pid] = 1
        else:
            pidDodMap[pid] = 0
    infd.close()

    print('Building pid-admission mapping, admission-date mapping')
    pidAdmMap = {}
    admDateMap = {}
    infd = gzip.open(admissionFile, 'rt', encoding='utf-8')
    header = infd.readline()

    print('admission header:',header)
    for line in infd:
        tokens = line.strip().split(',')
        pid = int(tokens[0])
        admId = int(tokens[1])
        admTime = datetime.strptime(tokens[2], '%Y-%m-%d %H:%M:%S')
        admDateMap[admId] = admTime
        if pid in pidAdmMap:
            pidAdmMap[pid].append(admId)
        else:
            pidAdmMap[pid] = [admId]
    infd.close()

    print('Building admission-dxList mapping')
    admDxMap = {}
    admDxMap_3digit = {}
    infd = gzip.open(diagnosisFile, 'rt', encoding='utf-8')
    header= infd.readline()
    print('diagnosis header:', header)
    for line in infd:
        tokens = line.strip().split(',')
        admId = int(tokens[1])
        dxStr = 'D_' + convert_to_icd9(tokens[3])
        dxStr_3digit = 'D_' + convert_to_3digit_icd9(tokens[3])

        if admId in admDxMap:
            admDxMap[admId].append(dxStr)
        else:
            admDxMap[admId] = [dxStr]

        if admId in admDxMap_3digit:
            admDxMap_3digit[admId].append(dxStr_3digit)
        else:
            admDxMap_3digit[admId] = [dxStr_3digit]
    infd.close()

    print('Building pid-sortedVisits mapping')
    pidSeqMap = {}
    pidSeqMap_3digit = {}
    pidSeqMap_3digit = {}
    for pid, admIdList in pidAdmMap.items():
        if len(admIdList) < 2: continue

        # فقط admissionهایی که در هر دو مپ وجود دارند
        valid_admissions = []
        for admId in admIdList:
            if admId in admDateMap and admId in admDxMap:
                valid_admissions.append(admId)
            else:
                print(f"Skipping admId {admId} - missing in maps")

        if len(valid_admissions) < 2: continue

        sortedList = sorted([(admDateMap[admId], admDxMap[admId]) for admId in valid_admissions])
        pidSeqMap[pid] = sortedList

        sortedList_3digit = sorted([(admDateMap[admId], admDxMap_3digit[admId]) for admId in valid_admissions])
        pidSeqMap_3digit[pid] = sortedList_3digit
    print('Building pids, dates, mortality_labels, strSeqs')
    pids = []
    dates = []
    seqs = []
    morts = []
    for pid, visits in pidSeqMap.items():
        pids.append(pid)
        morts.append(pidDodMap[pid])
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            seq.append(visit[1])
        dates.append(date)
        seqs.append(seq)

    print('Building pids, dates, strSeqs for 3digit ICD9 code')
    seqs_3digit = []
    for pid, visits in pidSeqMap_3digit.items():
        seq = []
        for visit in visits:
            seq.append(visit[1])
        seqs_3digit.append(seq)

    print('Converting strSeqs to intSeqs, and making types')
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in visit:
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
            newPatient.append(newVisit)
        newSeqs.append(newPatient)

    print('Converting strSeqs to intSeqs, and making types for 3digit ICD9 code')
    types_3digit = {}
    newSeqs_3digit = []
    for patient in seqs_3digit:
        newPatient = []
        for visit in patient:
            newVisit = []
            for code in set(visit):
                if code in types_3digit:
                    newVisit.append(types_3digit[code])
                else:
                    types_3digit[code] = len(types_3digit)
                    newVisit.append(types_3digit[code])
            newPatient.append(newVisit)
        newSeqs_3digit.append(newPatient)

    pickle.dump(pids, open(outFile + '.pids', 'wb'), -1)
    pickle.dump(dates, open(outFile + '.dates', 'wb'), -1)
    pickle.dump(morts, open(outFile + '.morts', 'wb'), -1)
    pickle.dump(newSeqs, open(outFile + '.seqs', 'wb'), -1)
    pickle.dump(types, open(outFile + '.types', 'wb'), -1)
    pickle.dump(newSeqs_3digit, open(outFile + '.3digitICD9.seqs', 'wb'), -1)
    pickle.dump(types_3digit, open(outFile + '.3digitICD9.types', 'wb'), -1)
    # run
    # cd D:\app\PythonProject\Paper-Implementation\dataset\MIMIC
    # python process_mimic.py RawData/admissions.csv.gz RawData/diagnoses_icd.csv.gz RawData/patients.csv.gz CleanData/mimic_output