from dateutil import parser

dt = parser.parse("2016-04-08 14:55:03")
dt1 = parser.parse("2016-04-08")
dt2 = parser.parse("14:55:03")

strList = list()
strList.append("")
strList.append(" ")
strList.append(None)
strList.append("2016-04-08 14:55:03")

for str in strList:
    if str is not None and bool(str.strip()):
        dt3 = parser.parse("")

test=1