'''
Created on Jan 3, 2024

@author: paepcke

This module contains two classes: ClassSerializingEncoder, 
and ClassSerializingDecoder. They supplement the 
standard json package, and interact with it.

Two ways to interact with this module:

    jstr      = json.to_json(<your_value>, cls=ClassSerializingEncoder)
    recovered = json.from_json(jstr, cls=ClassSerializingDecoder)

Or, more conveniently:

    jstr      = json_plus.to_json(<your_value)
    recovered = json_plus.from_json(jstr)
    
The module (de)serializes instances of user classes by
calling their to_json() and from_json() methods. A 
to_json() method is expected to return a string that, when
provided to the same class' from_json() will generate an 
an instance that is == to the original. 

Note: the from_json() method must be a classmethod, or 
      staticmethod, because no instance exists when it 
      is called.
      
Instances are serialized by placing them into the following
template string:

      ob__<YourClassNm>_"return-from-to_json()"__ob
      
The leading ob__ and trailing __ob may be changed by changing
the JSON_PREFIX and JSON_SUFFIX constants below.            
'''
import inspect
from common.general_utils import GeneralUtils
from common.serializable_enum import JSON_PREFIX, JSON_SUFFIX
import json
import re

#import builtins

# Given a JSON string generated from an instance
# from the encode() method, extract the class name 
# and payload. The class name will be in match.group(1). The
# payload will be in match.group(2):              cls-nm  payload
#                                                   V       V
#**** EXTRACTION_PATTERN = re.compile(fr'{JSON_PREFIX}<([^>]*)>_"(.*)"{JSON_SUFFIX}')
EXTRACTION_PATTERN = re.compile('ob__<([^>]*)>_(.*)__ob')

# Types that are enclosed in quotes when turned
# to JSON in isolation, but not when nested in 
# dicts or lists:
LITERALS_IN_NESTED = [str, int, float, type(None), bool]

# ------------------- Class ClassSerializingEncoder ------------------

class ClassSerializingEncoder(json.JSONEncoder):
    '''
    JSON serializer that json-encodes any instance
    of a class that provides a to_json() method.
    '''

    def encode(self, *args):
        '''
        Return the JSON form or *args. There may
        be either one, or two args. If one arg, it
        should be a primitive data type. Else it should
        be a pair [class, payload], where class is a 
        Class object, and payload is passed to that
        class' to_json() method.

        :raises TypeError: first of two given args is
            a string that is not a Class name. Or it
            is a Class object that is not found in the
            namespace
        :raises TypeError: first of two args is neither
            a string (class name), nor a Class object
        :raises ValueError: fewer than 1, or more than
            two args are passed in
        :raises TypeError: given arg(s) cannot be converted
            to JSON, either as an instance, nor as a primitive.
            Happens, for instance when a Class has no to_json()
            method.
        :return: a JSON encoded item
        :rtype: str
        '''
    
        if len(args) == 1:
            item = args[0]
            try:
                # Is value None, True, False?
                return self._primitive_to_json(item)
            except TypeError:
                # Not a special value
                pass
            cls_info      = None
            payload_info  = None
        elif len(args) == 2:
            item = None
            cls_info     = args[0]
            payload_info = args[1]
            if isinstance(cls_info, str):
                # cls_info is a class name:
                cls = GeneralUtils.class_named(cls_info)
                if cls is None:
                    raise TypeError(f"Class {cls_info} is unknown.")
            elif isinstance(cls_info, type):
                cls = cls_info
            else:
                err_msg = f"The cls_info arg must be a class name or class obj, not {cls_info}"
                raise TypeError(err_msg)
        else:
            err_msg = f"Number of args must be 1 or 2, not {len(args)}: {args}"
            raise ValueError()

        if item is not None:
            # We received one argument. Go through the
            # cases:
            
            if type(item) == str:
                # Arg gets an extra layer of quotes:
                jstr = json.dumps(item)
                return jstr
            
            elif type(item) in (int, float):
                # Numbers are turned into a string
                return str(item)
                
            elif GeneralUtils.has_method(item, 'to_json'):
                # Arg is an instance of a class that
                # provides a to_json() method. Use that:
                payload = item.to_json()
                jstr = self._encode_instance(item.__class__.__name__, payload)
                return jstr        
            
            elif type(item) == dict:
                # Encode each value, and have
                # keys be surrounded by double quotes:
                # Handle each key-value pair
                j_str_dict = {key : val 
                              if type(val) in LITERALS_IN_NESTED
                                else self.encode(val)
                              for key, val
                              in item.items()
                }
                # Dict is returned as a string:
                # Close the dict:
                j_res = json.dumps(j_str_dict)
                return j_res
            
            elif type(item) == list:
                # In lists, strings must be turned into 
                #   "str", while when encoded outside an
                # array, strings are encoded
                #   '"str"'
                # So: for arrays, handle each element, incl strings
                # ['foo', 'bar] => '["foo", "bar"]'
                j_str_list = [val 
                              if type(val) in LITERALS_IN_NESTED
                                else self.encode(val)
                              for val
                              in item
                ]
                # List is returned as string, after
                # removing the leading ", ":
                arr_str = json.dumps(j_str_list)
                return arr_str
        
        # We must have received two args: 
        # a class and a payload, so this
        # call must be the result of a call
        # to some instance's to_json():
 
        # Encode the object:
        try:
            jstr = self._encode_instance(cls, payload)
            return jstr
        except Exception as e:
            err_msg = (f"Cannot encode {args} either as instance "
                       f"or via built-in json "
                       f"(maybe object w/o to_json() method?): {e}"
            )
            raise TypeError(err_msg)
        
    #------------------------------------
    # default
    #-------------------
    
    def default(self, obj):
        '''
        Called by built-in json.encoder when it
        cannot serialize an item. We call to_json()
        on the object. If that succeeds, we use
        its return value. 
    
        :param obj: item on which json.encode() punted
        :type obj: any
        :return: json-serialized string
        :rtype: str
        '''
        super().default(obj)
        # # If obj is already an encoded
        # # instance, return it unchanged:
        # if self._is_encoded_instance(obj):
        #     return obj         
        # if inspect.ismethod(obj.to_json):
        #     return obj.to_json()
        #
        # super().default(obj)
        # try:
        #     magic_str = obj.to_json()
        #     instance_json_str = self._encode_instance(obj.__class__,
        #                                          magic_str)
        #     return instance_json_str
        # except AttributeError:
        #     # Will raise a type error:
        #     super().default(obj)

    #------------------------------------
    # _primitive_to_json
    #-------------------

    def _primitive_to_json(self, item):
        '''
        Converts the primitives None, True,
        and False to corresponding JSON values.
        If any value other than these is passed
        in, raises TypeError.

        :param item: value to convert
        :type item: Any
        :return: a JSON equivalent of item
        :rtype: str
        :raises TypeError when item is not special
        '''
        if item is None:
            return 'null'
        elif item is True:    # Don't use ==, b/c obj.__eq__
            return 'true'    # may fail.
        elif item is False:
            return 'false'
        raise TypeError(f"Item not a special value: {item}")

    #------------------------------------
    # _encode_instance
    #-------------------
    
    def _encode_instance(self, class_info, payload):
        '''
        Given an instance of any class that provides
        a to_json() method, return a serialized JSON 
        string for use by itself, or embedded in any 
        JSON structure. 
        
        Example1:
        
            class Foo:
            
               def __init__(self, blue_temp, red_temp, other=None):
                   self.blue_temp = blue_temp
                   self.red_temp  = red_temp
                   self.other     = other
               
               def to_json(self):
                   state = {'blue' : self.blue_temp, 
                            'red'  : self.red_temp,
                            'other': self.other
                            }
                   return state
            
               @classmethod
               def from_json(cls, jstr)
                   # The jstr will be similar to:
                   #   "{'blue' : 'cool', 'red' : 'hot', 'other' : None}"
                   # Recover the dict:
                   state = json_plus.from_json(jstr)
                   
                   obj = Foo(state['blue'], state['red'], state['other'])
                   
            Now use the class:
	            inst = Foo('cool', 'hot')
	            jstr = json_plus.dumps(inst)
            
            will return:
               'ob__<Foo>_"{'blue' : 'cool', 'red' : 'hot', 'other' : None}"__ob'
            
        Note that the states are processed recursively. For instance,
             
             foo_obj = Foo('cool', 'hot', <instance of SomeClass>)
             
        Assuming the SomeClass includes to_json() and from_json() methods,
        the above calls to json_plus.dumps() and json_plus.loads() will
        work as expected. That is, the SomeClass instance's encoded
        state will be embedded in the json string. Something like:
        
            'ob__<Foo>_"{'blue' : 'cool', \
                         'red' : 'hot', \
                         'other' : 'ob__<SomeClass>_"some-state"__ob' \
                         }"__ob'
              
        End users need not concern themselves with the json strings,
        other than for storage, or sending over a wire.
        
        Note that None is turned into null to fit into the JSON conventions.
              
        :param class_info: class object, or name of a class.
            The class should provide a from_json() method that
            the output of this method, and recreates the object
        :type class_info: {Class | str}
        :return: a string that is valid for use
            as a value in a JSON object
        '''

        # We pre-, and append special values to a member's
        # value, to make the value recognizable to our
        # decoder.
        # We also add the member's class name in angle 
        # brackets. For example, if myobj is an
        # instance of MyClass, and myobj.to_json() 
        # returns 'foobar', the json string will be:
        # 
        #   ob__<MyClass>_"foobar"__ob
        
        if type(class_info) == str:
            # Find the class object in the call stack:
            class_obj = GeneralUtils.class_named(class_info)
            if class_obj is None:
                raise ValueError(f"Cannot find definition of class '{class_info}'")
            class_nm = class_info
        else:
            # Arg class_info is a class object:
            class_nm = class_info.__name__

        return f'{JSON_PREFIX}<{class_nm}>_{payload}{JSON_SUFFIX}'


# ------------------- Class ClassSerializingDecoder ------------------

class ClassSerializingDecoder(json.JSONDecoder):
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self):
        super().__init__()

    #------------------------------------
    # decode
    #-------------------
    
    def decode(self, json_data):
        '''
        Given a string that is legal json, return 
        a reconstituted item from which ClassSerializingEncoder.encode()
        created the given json_data.
        
        This method handles decoding of any instances
        of classes that provide to_json() and from_json()
        which use the ClassSerializingEncoder.encode() to generate
        json.
        
        :param json_data: json string to decode
        :type json_data: str
        :return: reconstituted original: string/list/dict/instances
        '''

        # Special cases: input "null" ===> result of None:
        if json_data is None:
            return None 
        
        if json_data in ('null', '"null"'):
            return None
        
        if json_data in ('false', 'False', '"false"', '"False"'):
            return False
                    
        if json_data in ('true', 'True', '"true"', '"True"'):
            return True
        
        if type(json_data) in [int, float]:
            # Numbers eval to themselves
            return json_data

        if self._is_encoded_instance(json_data):
            # Find the class name and magic string
            # in the json_data:
            cls, payload = self._parse_encoded_instance(json_data)
            if not isinstance(cls, type):
                raise ValueError(f"Could not identify a class from {json_data}")
            inst = cls.from_json(payload)
            return inst

        else:
            try:
                new_dict = self._parse_encoded_dict(json_data)
                return new_dict
            except ValueError as e:
                # Json is not a parsable dict
                pass
            
            try:
                new_list = self._parse_encoded_list(json_data)
                return new_list
            except ValueError as e:
                # Json is not a parsable list:
                pass
            
            try:
                if type(json_data) == str:
                    res = super().decode(json_data)
                    return res
            except json.JSONDecodeError as e:
                # When strings are in an array, they
                # are stimply "foo". But if they are outside
                # of an array, they are '"foo"'. If this method
                # is called in the process of decoding an array,
                # the above super().decode() will throw a 
                # JSONDecodeError. In that case we return the
                # 'simple' string directly:
                return json_data
            except Exception as e:
                # Anything else: bad news:
                raise ValueError(f"Could not decode data: {e}.\nOffending item: {json_data}")

    #------------------------------------
    # _decode_element
    #-------------------
    
    def _decode_element(self, value):
        '''
        Given a value, if:
           o value is the JSON of a serializable user
             class' instance, the instance is re-generated
             and returned
           o value is a JSON string, it is loaded via
             json_utils.from_json(), and returned.
           o if value is None, return None
           o if value is a number it is returned unchanged
        
        :param value: value to convert
        :type value: str
        :return: instance of user defined class, or
            any other value decoded from the given
            value.
        :rtype: Any
        '''

        if value is None:
            return None
        elif type(value) in (int, float):
            return value
        
        # Try to turn the value string into 
        # the member of a subclass of 
        elif self._is_encoded_instance(value):
            try:
                class_obj, magic_val = self._parse_encoded_instance(value)
                instance = class_obj(magic_val)
                return instance
            except (TypeError, ValueError):
                raise ValueError(f"Bad value: {value} for JSON decoding.")

        else:
            # str() is in case value is a builtin,
            # such as an int. Loads would fail, so
            # turn into a str first. I value came in
            # as a str, the str() does nothing:
            try:
                json_str = json.dumps(value)
            except TypeError:
                json_str = json.dumps(str(value))
            return json_str
        
    #------------------------------------
    # _is_encoded_instance
    #-------------------
    
    def _is_encoded_instance(self, json_str):
        '''
        Returns True if given json_str encodes
        an enum member, and was created by the
        SerializableEnum.to_json() method. Else
        returns False.
        
        We recognize both:
           '<JSON_PREFIX>...<JSON_SUFFIX>'
        and:
           '"<JSON_PREFIX>...<JSON_SUFFIX>"'
        
        :param json_str: json string to examine
        :type json_str: str
        :returns: whether or not the json_str is 
            a json-encoded enum member
        :rtype bool
        '''
        # Remove possibly enclosing double quotes.
        # The try/except is needed when non-strings
        # are passed in; those have not strip() method:
        # try:
        #     json_str = GeneralUtils.strip_quotes(json_str)
        # except TypeError:
        #     pass
        
        if type(json_str) != str:
            return False
        match = EXTRACTION_PATTERN.match(json_str)
        return False if match is None else True
 
    #------------------------------------
    # _parse_encoded_dict 
    #-------------------
    
    def _parse_encoded_dict(self, jstr):
        '''
        Takes a dict that is enclosed in a string,
        and whose values are json-encoded. Tries to 
        return a dict with all values decoded.
        
        Ex1:    "{'foo' : 10, 'bar' : 20}"
        returns {'foo' : 10, 'bar' : 20}
        
        Ex2:    '{"my_inst" : "ob__<MyClass>_\"magic string\"__ob"}'
        returns {'my_inst : <Instance of MyClass>}
        
        Raises ValueError if parsing fails.
        
        :param jstr: string to parse
        :type jstr: str
        :return recursively decoded dict
        :rtype dict
        :raise ValueError if parsing fails
        '''
        
        if jstr.__class__ != str:
            raise TypeError(f"zArgument must be a string, not {jstr}")

        if not jstr.startswith('{') \
           or not jstr.endswith('}'):
            raise ValueError(f"Decode: not a dict: {jstr}")
        else:
            try:
                # Safely eval the jstr, and check whether
                # result is a dict:
                maybe_dict = json.loads(jstr)
            except Exception:
                raise ValueError(f"Failed to json.loads during dict decoding. Offender: {jstr}")

        if type(maybe_dict) != dict:
            raise ValueError(f"Failed to eval {jstr} during dict decoding")
        
        try:
            new_dict = {}
            for key, val in maybe_dict.items():
                if self._is_encoded_instance(val):
                    processed_val = self.decode(val)
                elif type(val) in LITERALS_IN_NESTED:
                    processed_val = val 
                else:
                    processed_val = self.decode(val)
                new_dict[key] = processed_val

        except Exception as e:
            raise ValueError(f"Dict decoding failed for {jstr} for value {val}: {e}")

        return new_dict

    #------------------------------------
    # _parse_encoded_list
    #-------------------
    
    def _parse_encoded_list(self, jstr):
        '''
        Takes a list that is enclosed in a string,
        and whose elements are json-encoded. Tries to 
        return a list with all values decoded.
        
        Ex1:    "['foo', 20]"
        returns ['foo', 20]
        
        Ex2:    '[10, "ob__<MyClass>_\"magic string\"__ob"]'
        returns [10, <Instance of MyClass>}]
        
        Raises ValueError if parsing fails.
        
        :param jstr: string to parse
        :type jstr: str
        :return recursively decoded dict
        :rtype list
        :raise ValueError if parsing fails
        '''
        
        # Safely eval the jstr, and check whether
        # result is a list:
        if not jstr.startswith('[') \
           or not jstr.endswith(']'):
            raise ValueError(f"Decode: not a list: {jstr}")
        else:
            try:
                maybe_list = json.loads(jstr)
            except Exception:
                raise ValueError(f"Could not eval string to a dict: {jstr}")

        if type(maybe_list) != list:
            raise ValueError(f"Failed to eval {jstr} during list decoding")
        
        try:
            new_list = []
            for el in maybe_list:
                if self._is_encoded_instance(el):
                    processed_el = self.decode(el)
                elif type(el) in LITERALS_IN_NESTED:
                    processed_el = el 
                else:
                    processed_el = self.decode(el)
                new_list.append(processed_el)
        except Exception as e:
            raise ValueError(f"List decoding failed for {jstr}: {e}")
        return new_list

    #------------------------------------
    # _parse_encoded_instance 
    #-------------------
    
    @staticmethod
    def _parse_encoded_instance(jstr):
        '''
        From something like:
    
          'ob__<MsgUUID>_"c082a34c94f84d4daf10273c6157191f"__ob'
        
        return (MsgUUID, "c08...")
        where MsgUUID is a class object.
        
        :param jstr: string that encodes a class
        :type jstr: str
        :param incl_dbl_quotes: if True, the usually surrounding
            double quotes around the payload are included.
            Else they are stripped.
        :type incl_dbl_quotes: bool
        :return: (<class>, payload)
        :rtype (Class, str)
        :raise ValueError if pattern does not match
        :raise TypeError if class name in json string
            is not in the class registry.
        '''
        
        # Truly a string?
        if jstr.__class__ != str:
            raise TypeError(f"Argument must be a string, not {jstr}")
        
        match = EXTRACTION_PATTERN.match(jstr)
        if match is None:
            raise TypeError(f'String "{jstr}" cannot be parsed as an encoded instance')
        cls_nm, payload = match.groups()
        try:
            cls = GeneralUtils.class_named(cls_nm)
        except (KeyError, NameError):
            raise TypeError(f"Class {cls_nm} from {jstr} is unknown")

        # If the class is int or float, pull them out of 
        # the payload to return the actual value:
        
        if cls in (int, float, dict):
            payload = json.loads(payload)
        
        #******* Decode Payload        
        # Try to decode the payload:
        # try:
        #     payload_decoded = json.from_json(payload)
        #     # If payload is already the instance of a class,
        #     # we are done. But if the payload is something
        #     # like a dict, must call cls' from_json() to 
        #     # get the object:
        #     if cls_nm not in ('int', 'float', 'type', 'dict'):
        #         payload_decoded = cls.from_json(payload_decoded)
        # except json.JSONDecodeError as _e:
        #     # Maybe it's one of our own classes, 
        #     # which knows how to deal:
        #     try:
        #         payload_decoded = cls.from_json(payload)
        #     except (AttributeError, TypeError, ValueError):
        #         # Or the result item is to be the member
        #         # of a serializable enum, in which case
        #         # the class would respond to from_value():
        #         if issubclass(cls, SerializableEnum):
        #             payload_decoded = cls.from_value(payload)
        #         else:
        #             payload_decoded = payload
        #return (cls, payload_decoded)
        #******* End Decode Payload
        return (cls, payload)
    
    #------------------------------------
    # _is_simple_str
    #-------------------

    def _is_simple_str(self, s):
        '''
        Returns True if given string is not 
        enclosed in double quotes. Used to 
        decide whether to pass a string into
        json.from_json(), or treat it as a string
        that does not need to be decoded.
        
        However, if the string is readable as
        a number, return False 
        
        :param s: string to examine
        :type s: str
        :return: whether or not the string is
            enclosed in double quotes
        :rtype: bool
        '''

        # Not even a st?
        if type(s) != str:
            return False
        
        # A stringified number?
        try:
            GeneralUtils.parse_number(s)
            # Parsing succeeded, so s is 
            # stringfied num:
            return False
        except ValueError:
            pass
                        
        # Enclosed in double quotes?
        if s.startswith('"') and s.endswith('"'):
            # String enclosed in double quotes:
            return False
        # Enclosed in curly braces?
        if s.startswith('{') and s.endswith('}'):
            # String enclosed in curlies
            return False
        # Enclosed in brackets?
        if s.startswith('[') and s.endswith(']'):
            # String enclosed in brackets:
            return False
        
        #legal_chrs_pat = re.compile(r'^[\w+\d+\s+]*$')
        #legal_chrs_only = legal_chrs_pat.search(s)
        #return legal_chrs_only
        return True


# -------------------- to_json and from_json conveniences ---------

def _safe_conversion(item):
    try:
        safe_val = json.dumps(item)
    except TypeError:
        safe_val = dumps(item, standardize=False)
    return safe_val        

#------------------------------------
# to_json 
#-------------------

def dumps(*args):
    '''
    Return a json string that can be decoded via
    from_json() (both in this json_plus module).
    
    In addition to the built-in json.dumps(), this
    method handles any class that implements to_json(),
    which creates standard json strings like this:
    
      'ob__<ClassName>_"payload-for-from_json()"__ob'
    
    :param args: either a single item to serialize, or,
        when called from an instance's to_json() method,
        a class object and a payload to use for
        reconstituting the calling object with from_json()
        later.
    :type item: {Any | ({class-obj | class_name}, Any}
    :return: a string that can be used for from_json() to recover
        the original item.
    :rtype: str
    :raise ValueError
    '''
    res = encoder.encode(*args)
    return res

#------------------------------------
# from_json
#-------------------

def loads(json_str):
    
    res = decoder.decode(json_str)
    return res

# ------------------ Encoder/Decoder Instances ---------

encoder = ClassSerializingEncoder()
decoder = ClassSerializingDecoder()