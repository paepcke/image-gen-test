'''
Created on Dec 30, 2023

@author: paepcke

NOTE: unittests for this module are in test_json_plus.py

This module provides three facilities:
    o A subclassable class SerializableEnum, which 
      may be json serialized, and deserialized
      using 
      
          json_plus.dumps(<instance of SerializableEnum subclass>)
          
      The reverse is:
      
          json_plus.loads(<json created by json_plus.dumps())
          
    o Methods
    
         to_json()     # called by json_plus.dumps()
         from_json()   # called by json_plus.loads()
         
Example:
        class MyEnum(SerializableEnum):
           FOO = 10
           BAR = 20
            
        json_str = json_plus.dumps(MyEnum.BAR)
        member   = json_plus.loads(json_str)
        assert(member == MyEnum.BAR)
        
Enum subclass members may be values in dicts.

Example:
        my_dict  = {'foo' : 10, 'bar' : MyEnum.FOO}
        json_str = json_plus.dumps(my_dict)
        orig     = json_plus.loads(json_str)
        assert   = orig == my_dict         
      
'''

from common.general_utils import (
    GeneralUtils)
from enum import (
    Enum)
import re

# Prefix and suffix strings used in Enum 
# json encoding. These must be module-global,
# because they are used in subclasses of Enum.
# Subclassing of Enum base classes is only allowed
# if the base class has no attributes: 

JSON_PREFIX = 'ob__'
JSON_SUFFIX = '__ob'

class SerializableEnum(Enum):
    '''
    Subclasses inherit conveniences for Enum
    structures, such as finding members that 
    corresponding a a given name, or value. 
    The class also provides facilities for turning
    any member into a JSON-acceptable string, and
    to recover a member from such a string.
    
    Terms: in
    
         class MyEnum(SerializableEnum):
            FOO = 10
            BAR = 20
    
    'FOO' is called a 'name', and 10 is called a 'value'
    '''

    #------------------------------------
    # names
    #-------------------
    
    @classmethod
    def names(cls):
        '''
        Returns the names of enum members:
        :return: list of all enum member names
        :rtype: (str) 
        '''
        return list(map(lambda c: c.name, cls))
    
    #------------------------------------
    # values
    #-------------------

    @classmethod
    def values(cls):
        '''
        Returns the value of each enum member:
        
        :return: list of all enum member values
        :rtype: (any) 
        '''
        return list(map(lambda c: c.value, cls))
    
    #------------------------------------
    # to_json
    #-------------------
    
    def to_json(self):
        '''
        Given an instance of a subclass, return a serialized
        string for use in a JSON structure. The resulting
        string may be turned back into the corresponding
        enum self via the from_json() method.
         
        The calling method in json_plus will surround the
        returned string with a pre- and postfix:
        
            <PREFIX>_<EnumClassName>_"member_name"_<SUFFIX>
        
        Example:
        
            class MyEnum(SerializableEnum):
               FOO = 10
               BAR = 20

            MyEnum.FOO.to_json()
              returns: "ob__<MyEnum>_"FOO"__ob"
         
        :param self: enum member
        :type self: SerializableEnum
        :return: a string that is valid for use
            as a name in a JSON object
        '''
        jstr = self.name
        return jstr
        
    #------------------------------------
    # from_json
    #-------------------
    
    @classmethod
    def from_json(cls, member_jstr):
        '''
        Given a JSON string that was created by the to_json()
        method of an enum subclass, return a member of the
        enum subclass. Ex.:
        
            class MyEnum(SerializableEnum):
               FOO = 10
               BAR = 20

            given json_str: "BAR"
            return MyEnum.BAR
        
        For historical reasons, be tolerant of being handed a full 
        json encoding, such as 'ob__<MyEnum>_"BAR"__ob'
             
        :param member_jstr: json digestible string created by to_json() 
        :type member_jstr: str
        :param str2nums: whether or not to assume that strings
            that can be converted to numbers should be assumed
            to be intended as numbers, or left as strings.
            Default is to convert to numbers.
        :type str2nums: bool
        :return a member of a subclass
        :rtype SerializableEnum
        :raise TypeError if given string was not
            encoded with this class' to_json()
        :raise ValueError if string was created
            by a call to to_json(), but the name is
            incorrect in some way.
        '''
        # Cover jstr being JSON for None:
        if member_jstr == 'null':
            return None
        
        # From something like:
        #    ob__<ModelName>_"ADA"__ob
        # extract "ADA":
        pat = re.compile(r'[^"]*"([^"]*)"__ob')
        match = pat.search(member_jstr)
        if match is None:
            # Try for member_jstr being the name of the enum:
            try:
                member = cls.from_name(member_jstr)
                return member
            except Exception as e:
                msg = f"String {member_jstr} not recognized: {e}"
                raise TypeError(msg)
        member = cls.from_name(match.group(1))
        return member
    
    #------------------------------------
    # from_value
    #-------------------
    
    @classmethod
    def from_value(cls, member_value, str2num=True):
        '''
        Given a subclass object of SerializableEnum,
        and one of the member values (i.e. right sides
        of the enum), return an enum member. Ex.:
        
           class MyEnum(SerializableEnum):
               FOO = 10
               BAR = 20
               
            MyEnum.from_value(20) --> MyEnum.BAR
        or: MyEnum.from_value('20') --> MyEnum.BAR
        
        :param member_value: the member_value part of an enum member
        :type member_value: any
        :rtype SerializableEnum
        :raised ValueError if no respective member of the enum cls exists. 
        '''
        
        if str2num:
            # Try again with double quotes removed:
            try:
                member_value = GeneralUtils.strip_quotes(member_value)
            except (AttributeError, TypeError):
                # The member_value isn't a string,
                # so has no 'strip()' method:
                pass 
             
        # See whether the given str is a stringified number:
        try:
            numeric_value = GeneralUtils.parse_number(member_value)
        except ValueError:
            # Magic string is not a stringified number,
            # or a number string with extra quotes around
            # it:
            numeric_value = None
        
        for member in cls:
            if member.value == member_value:
                return member
            # Try to interpret the string as a number:
            else:
                if numeric_value is not None and member.value == numeric_value:
                    return member
            # Next enum member:
            continue 

        raise ValueError(f"'{member_value}' is not from a known {cls.__name__}")
    
    #------------------------------------
    # from_name
    #-------------------

    @classmethod
    def from_name(cls, member_name):
        '''
        Given a subclass object of SerializableEnum,
        and one of the member names (i.e. left sides
        of the enum member assignments), return an 
        enum member. Ex.:
        
           class MyEnum(SerializableEnum):
               FOO = 10
               BAR = 20
               
            MyEnum.from_name('BAR') --> MyEnum.BAR
        
        :param value: the name part of an enum member
        :type value: str
        :rtype SerializableEnum
        :raised ValueError if no respective member of cls exists. 
        '''
        
        if type(member_name) != str:
            raise TypeError(f"Name must be a string, not {type(member_name)}")
        
        # All name parts of enum members:
        name_list = cls.names()
        
        #*****name = member_name.upper()
        if member_name not in name_list:
            raise ValueError(
                f"Name must be one of {name_list}, not {member_name}")
        
        # Now we know that we have a proper name:
        for member in cls:
            if member.name == member_name:
                return member

        raise ValueError(f"'{member_name}' is not from a known {cls.__name__}")
