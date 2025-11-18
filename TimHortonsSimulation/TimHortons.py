class Order:
    # Intialize an order object
    def __init__(self, order_id, type, num_customers, num_items):
        self.order_id = order_id           # Unique identifier attached to child food items as well
        self.num_customers = num_customers # Number of customers for seating, irrelevant to drive-thru
        self.num_items = num_items         # Number of customers for checking condition of order completion
        self.items_completed = 0           # Number of items completed, must equal num_items to complete order

        # Walk-in, pick-up, or drive-thru
        if (type.lower() == "walk-in") or (type.lower() == "drive-thru") or (type.lower() == "mobile"):
            self.order_type = type.lower()
        else:
            # Prevent bugs
            raise NameError("Invalid order type string ", type.lower())


class Food:
    # Initialize food object, each order has many corresponding food items
    def __init__(self, food_id, order_id, type):
        self.food_id = food_id   # Id for food item
        self.order_id = order_id # Corresponding order
        self.food_type = type    # Type of food item (i.e. coffee, espresso, )

        # Type of food item (coffee, espresso, donut, panini, hashbrown, sandwich, etc.)
        if (type.lower() == "coffee") or (type.lower() == "espresso") or (type.lower() == "donut") or (type.lower() == "panini") or (type.lower() == "hashbrown") or (type.lower() == "sandwich"):
            self.food_type = type.lower()
        else:
            # Prevent bugs
            raise NameError("Invalid food type string ", type.lower())
    

class Staff:
    # Create a staff worker
    def __init__(self, staff_id, type):
        self.staff_id = staff_id # Staff unique identifier
        self.staff_idle = True   # Bool for if the staff is working or not

        # Worker type (i.e. barista, cashier, kitchen, drive-thru window, etc.)
        if (type.lower() == "barista") or (type.lower() == "cashier") or (type.lower() == "kitchen") or (type.lower() == "window"):
            self.staff_type = type.lower()
        else:
            # Prevent bugs
            raise NameError("Invalid food type string ", type.lower())
    
    

class Equipment:
    # Create a piece of equipment
    def __init__(self, eq_id, type, num_slots):
        self.eq_id = eq_id         # Unique identifier for a specific piece of equipment
        self.eq_type = type        # The equipment type
        self.num_slots = num_slots # The total number of people who can work at a station/equipment
        self.used_slots = 0        # The number of equipment slots currently in use

        # Equipment type (i.e. panini-press, sandwich-station, coffee-maker, coffee-urn, etc.)
        if (type.lower() == "panini-press") or (type.lower() == "sandwich-station") or (type.lower() == "coffee-maker") or (type.lower() == "coffee-urn") or (type.lower() == "cash-register"):
            self.eq_type = type.lower()
        else:
            # Prevent bugs
            raise NameError("Invalid food type string ", type.lower())

