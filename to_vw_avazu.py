from datetime import datetime

def csv_to_vw(loc_csv, loc_output, train=True):
    start = datetime.now()
    print("\nTurning %s into %s. Is_train_set? %s"%(loc_csv,loc_output,train))
    i = open(loc_csv, "r")
    j = open(loc_output, 'wb')
    counter=0
    with i as infile:
        line_count=0
        for line in infile:

            # to counter the header
            if line_count==0:
                line_count=1
                continue

            # The data has all categorical features
            categorical_features = ""
            counter = counter+1


            line = line.split(",")
            if train:
                #working on the date column. We will take day , hour
                a = line[2]
                new_date= datetime(int("20"+a[0:2]),int(a[2:4]),int(a[4:6]))
                day = new_date.strftime("%A")
                hour= a[6:8]
                categorical_features += "hr_%s " % hour
                categorical_features += "day_%s " % day
                # 24 columns in data    
                for i in range(3,24):
                    if line[i] != "":
                        categorical_features += "c%s_%s " % (str(i),line[i])
            else:
                a = line[1]
                new_date= datetime(int("20"+a[0:2]),int(a[2:4]),int(a[4:6]))
                day = new_date.strftime("%A")
                hour= a[6:8]
                categorical_features += "hr_%s " % hour
                categorical_features += "day_%s " % day
                for i in range(2,23):
                    if line[i] != "":
                        categorical_features += "c%s_%s " % (str(i+1), line[i])


            if train: #we care about labels
                if line[1] == "1":
                    label = 1
                else:
                    label = -1  #we set negative label to -1
                record = "%s | %s" % (label, categorical_features)

            else: #we dont care about labels
                record = "1 | %s" % (categorical_features)

            j.write(record.replace('\t', ''))

            if counter % (1 * 10 ** 6) == 0:
                print("%s\t%s"%(counter, str(datetime.now() - start)))

            if counter % (10 * 10 ** 6) == 0:
                break

    print("\n %s Task execution time:\n\t%s"%(counter, str(datetime.now() - start)))

csv_to_vw("./ctrdata/avazu/train.csv", "./ctrdata/avazu/train.vw",train=True)
csv_to_vw("./ctrdata/avazu/test.csv", "./ctrdata/avazu/test.vw",train=False)