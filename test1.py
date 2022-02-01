from test_class import Airport
from test_class import Chinese_Airport
# import pandas as pd
# import numpy as np
# s = pd.Series([1, 3, 5, np.nan, 6, 8])
# print(s)
# print("Hello World")
# for i in range(1,10):
#     i = i+1
# print(i)

# cordinate_list=[(1,0),(1,1),(2,9)]
# print(cordinate_list)
# print(cordinate_list[0])
# print(cordinate_list[2][1])
# print(type(cordinate_list))
# print(type(cordinate_list[1]))
# print(type(cordinate_list[1][0]))
# cordinate_list[0]=(100,100)
# print(cordinate_list)
# copy_cord = cordinate_list.copy()
# print(copy_cord)
# # cordinate_list.extend((7,23))
# cordinate_list.append((8,30))
# cordinate_list.append({"Jan":1,"Feb":2})
# print(type(cordinate_list))
# print(cordinate_list)
# print(type(cordinate_list[4]))
# dic_cor = cordinate_list[4]
# print(dic_cor.get("Jan"))
# print(dic_cor.keys())
# print(dic_cor.items())
# key_list = list(dic_cor.keys())
# val_list = list(dic_cor.values())
# print(key_list)
# print(val_list)
# ind = val_list.index(int(input("Input Number: ")))
# print(key_list[ind])

Airport_list = []
Airport_Dic = {}
Customized_Airport = int(input("How many airports need to be added?"))
for i in range(Customized_Airport):
    AP_Code = input("Please input the airport code")
    # if the AP_Code exist:
    # if is the hub airport, add new hbu airline, else for the next new airport
    if(AP_Code in Airport_Dic.keys()):
        new_hub_airline = input("Airport Exist and Is there any update for the Hub Airlines?Y/N")
        if(new_hub_airline.lower() =="y"):
            # add new hub airline
            print("update Hub Airline Info")
            Hub_Airline = input("Please input the Hub Airline:")
            # Airport_Dic[AP_Code].add_hub(Hub_Airline)
            Airport_Dic[AP_Code].add_hub_list(Hub_Airline)
    else:
        AP_Name = input("Please input the airport name")
        AP_Lc = input("Please input the airport location")
        AP_Hub = input("Is the Hub for the Airlines? Y/N")
        AP_CN= input("Is the Chinese Airport?Y/N")
        if(AP_CN.lower()=="y"):
            New_Ap = Chinese_Airport(AP_Code,AP_Name,AP_Lc)
            PY=input("Please input the PINYIN for the Airport")
            New_Ap.get_pinyin(PY)
        else:
            New_Ap = Airport(AP_Code,AP_Name,AP_Lc)
        if AP_Hub.lower() == "y":
            Hub_Airline = input("Please input the Hub Airline:")
            New_Ap.add_hub(Hub_Airline)
        temp_air_port_dict = {AP_Code: New_Ap}
        Airport_Dic.update(temp_air_port_dict)
        temp_air_port_dict.clear()
        Airport_list.append(New_Ap)
for item in Airport_Dic:
    # if(hasattr(Airport_Dic[item],"hub")):
    # if (hasattr(Airport_Dic[item], "hub_list")):
    if (len(Airport_Dic[item].hub_list)>0):
        if (hasattr(Airport_Dic[item], "pinyin")):
            # print(item + ":" + Airport_Dic[item].airport_name + " and is the Hub for " + Airport_Dic[item].hub+", PINYIN:" +Airport_Dic[item].pinyin)
            #print(item + ":" + Airport_Dic[item].airport_name + " and is the Hub for " + Airport_Dic[item].hub_list[0]+", PINYIN:" +Airport_Dic[item].pinyin)
            print(item + ":" + Airport_Dic[item].airport_name + " and is the Hub for " + Airport_Dic[item].hub_list+ ", PINYIN:" + Airport_Dic[item].pinyin)
        else:
            # print(item + ":" + Airport_Dic[item].airport_name + " and is the Hub for " + Airport_Dic[item].hub)
            # print(item + ":" + Airport_Dic[item].airport_name + " and is the Hub for " + Airport_Dic[item].hub_list[0])
            print(item + ":" + Airport_Dic[item].airport_name + " and is the Hub for ",end = " ")
            print(Airport_Dic[item].hub_list,end = " ")
    else:
        if (hasattr(Airport_Dic[item], "pinyin")):
            print(item + ":" + Airport_Dic[item].airport_name + " and is not the Hub for any airlines, PINYIN:" + Airport_Dic[item].pinyin)
        else:
            print(item + ":" + Airport_Dic[item].airport_name+" and is not the Hub for any airlines")

# f= open("test.txt")
# print(f.readline())

# def Airline_Hub_Airport(airline,airport_dic):
#     Hub_Airport = []
#     for item in airport_dic:
#         if hasattr(airport_dic[item],"hub") and (airport_dic[item].hub==airline):
#             Hub_Airport.append(item)
#     if len(Hub_Airport) == 0:
#         print(airline+" has no Hub Airport")
#     else:
#         print(airline+"'has "+str(len(Hub_Airport))+" Hub Airport(s):")
#         for Airport_item in Hub_Airport:
#             print(Airport_item,end = " ")

def Airline_Hub_Airport(airline,airport_dic):
    Hub_Airport = []
    for item in airport_dic:
        for hub_airline in airport_dic[item].hub_list:
            if hub_airline==airline:
                Hub_Airport.append(item)
    if len(Hub_Airport) == 0:
        print(airline+" has no Hub Airport")
    else:
        print(airline+"'has "+str(len(Hub_Airport))+" Hub Airport(s):")
        for Airport_item in Hub_Airport:
            print(Airport_item,end = " ")
airline = input("Please input the Airline for Hub Airport Search")

Airline_Hub_Airport(airline,Airport_Dic)

