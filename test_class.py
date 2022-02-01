# Create the Airport Class
class Airport:
    # inintial attr, set the initial value when create it, and set the hub_list as the null list for future use
    def __init__(self, airport_code, airport_name,airport_locations):
        self.airport_code = airport_code
        self.airport_name = airport_name
        self.airport_locations = airport_locations
        self.hub_list = []
    # add hub to the airport, not use anymore
    def add_hub(self,hub):
        self.hub = hub
    # add hub airline to airport, could includes multiple airline for the same airport
    def add_hub_list(self, hub):
        self.hub_list.append(hub)
    # display detial info the airport
    def get_info(self):
        print("Airport infor: /n")
        print("The "+self.airport_code+": ")
        print("Name: "+self.airport_name+"/n Location: "+self.airport_locations)
# Son class inheriate from the father class (Airport), has all the attr and method and one unique attr : PINYIN
class Chinese_Airport(Airport):
    def get_pinyin(self,pinyin):
        self.pinyin = pinyin

