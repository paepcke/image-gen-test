'''
Created on Jan 2, 2024

@author: paepcke
'''
import random
from _datetime import (
    timedelta,
    datetime)
from contextlib import (
    contextmanager)
from pathlib import (
    Path)
import inspect
import os
import pathlib
import re
import sys
import json

# -------------------- ServerException ---------- 
class ServerExc(Exception):
    '''
    Given exception information from the server,
    print the exception. The exc_info is expected
    as a dict:
       {'exc_class': <name of exception class>,
        'exc_msg'  : <short descr of error>,
        'exc_tb'   : <traceback stack as a string>
        }
    Prints something like:
    
       Server side: AttributeError
       in server_side_func1
           my_var = 10
           server_side_file1 line 430
       in server_side_func2
           print(arg)
           server_side_file2 line 4
             ...
       
    '''
    def __init__(self, exc_info):
        super().__init__()
        self.exc_info = exc_info
        
    def __str__(self):
        lines = []
        lines.append(f"Server side: {self.exc_info['exc_class']}")
        lines.extend(self.exc_info['exc_tb'].split('\n'))
        return '\n'.join(lines)

# -------------------- Permission Exception Flavors ---------- 

class PermissionUnknownToken(PermissionError):
    pass
class PermissionIncorrectPwd(PermissionError):
    pass
class PermissionUnknownUser(PermissionError):
    pass
    

# -------------------- GeneralUtils ---------- 
class GeneralUtils:
    
    #------------------------------------
    # class_name
    #-------------------

    @staticmethod
    def class_name(inst):
        return inst.__class__.__name__

    #------------------------------------
    # class_named
    #-------------------
    
    @staticmethod
    def class_named(class_nm,
                    verification_callback=lambda _class_obj: True):
        '''
        NOTE: a simpler method might be:
                locals()[class-name-str]
                
              though equivalence to this expression, and
              the more thorough, dynamic one below is
              untested. 
        
        Given the class name of a user defined class,
        return its class object. The class may have been
        defined in modules up the call stack. We'll find
        it. If no class is found, the class name could 
        denote a builtin type, such as 'str'. In that case
        we return None
        
        :param class_nm: name of class to be found
        :type class_nm: str
        :return: class object whose name is class_nm
        :rtype: Class
        '''
        
        # A built-in type?
        builtins = {
            'int'    : int,
            'float'  : float,
            'dict'   : dict,
            'bool'   : bool,
            'list'   : list,
            'type'   : type
            }
        try:
            return builtins[class_nm]
        except KeyError:
            # Class name does not name a built-in type:
            pass
        
        frame = inspect.currentframe()
        while True:
            try:
                frame_locals  = frame.f_locals
                frame_globals = frame.f_globals
            except AttributeError as _e:
                # It's a built-in, or no such class exists:
                return None
            if class_nm in frame_locals.keys():
                enum_class = frame_locals[class_nm]
                if verification_callback(enum_class):
                    return enum_class
            elif class_nm in frame_globals.keys():
                enum_class = frame_globals[class_nm]
                if verification_callback(enum_class):
                    return enum_class
            
            try:
                frame = frame.f_back
            except Exception:
                # Reached top of call stack:
                return None

    #------------------------------------
    # deactivate_instance_methods
    #-------------------

    @staticmethod
    def deactivate_instance_methods(instance, exceptions=[]):
        '''
        Takes the instance of a class, and replaces each
        method implementation of that instance with a lambda
        that raises a NotImplementedError.

        NOTE: the implementations of any new, or other existing
        instances' methods remain unchanged.
        NOTE: classmethod are NOT affected

        :param instance: the instance whose methods 
            are to be deactivated
        :type instance: Object
        :param exceptions: optional list of method names that 
            are not to be replaced, defaults to []
        :type exceptions: list[str], optional
        '''

        # A helper function used in the lambda below
        def raise_exception(exception):
            raise exception
        
        cls = type(instance)  # Get the class of the instance
        # Collect a map <method-name> : <method-object>
        # of all the methods in the given instance's class,
        # safe those listed in exceptions:
        methods = {member[0]: member[1] 
                   for member 
                   in inspect.getmembers(cls, predicate=inspect.isfunction)
                   if member[0] not in exceptions
                   }

        for method_name, method_obj in methods.items():
            # Replace the method in the instance's dictionary
            setattr(instance, 
                    method_name, 
                    lambda *args, **kwargs: raise_exception(NotImplementedError(f"Method '{method_name}' deactivated")))

    #------------------------------------
    # _parse_number
    #-------------------

    @staticmethod
    def parse_number(a_string):
        '''
        Given a string that is either:
            o a positive or negative integer
            o a positive or negative float
        return an int or float, respectively,
        else raise ValueError.
        
        Note: raises ValueError for values like '3**3'
        
        :param a_string: string to be converted to a number
        :type a_string: str
        :return: the string converted to a number
        :rtype: {int | float}
        :raise ValueError: if string cannot be converted to a number
        '''
        
        try:
            num = int(a_string)
            return num
        except ValueError:
            # wasn't a pure positive or negative int
            pass 
        if a_string.translate({ord(i): None for i in '-.'}).isnumeric():
            num = float(a_string)
            return num
        # Could be exponent: 3**3 or 3*3:
        raise ValueError(f"String '{a_string}' is not a number")

    #------------------------------------
    # safe_eval 
    #-------------------
    
    @staticmethod
    def safe_eval(str_to_eval):
        res = eval(str_to_eval,
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )
        return res

    #------------------------------------
    # find_long_match
    #------------------- 
    
    @staticmethod
    def find_match_in_list(prefix, string_list, by='shortest'):
        '''
        Given a string and a list of strings, find the 
        longest or shortest string in string_list that starts with
        prefix.

        :param prefix: string to lead the result
        :type prefix: str
        :param string_list: list of strings to match against
        :type string_list: list(str)
        :param by: whether to find the longest or the shortest
            match. Options are 'shortest', and 'longest'
        :type by: str
        :return: the longest element of string_list that
            starts with prefix. Empty string if no match
        :rtype: _type_
        '''
        # Gather candidates:
        matches = [s for s in string_list if s.startswith(prefix)]
        # Find longest or shortest match, or empty string if none:
        if by == 'longest':
            return max(matches, key=len, default='')
        elif by == 'shortest':
            return min(matches, key=len, default='')
        else:
            raise TypeError(f"The 'by' kwarg must be 'shortest' or 'longest', not {by}")

    #------------------------------------
    # find_max_overlap
    #-------------------    

    def find_max_overlap(subject_str, choices):
        '''
        Given a string (subject_str), and a list of other 
        strings (choices), find the string in choices that 
        has the maximum overlap with subject_str.

        Example:
            subject_str = 'gpt-4o-mini-2024-07-18'
            choices     = ['gpt-4o', 'gpt-4o-mini']

            returns 'gpt-4o-mini

        If any choice exactly matches subject_str, then that
        choice is returned. In the example above, if
        subject_str is 'gpt-4o', then that string is returned,
        not gpt-4o-mini.

        :param subject_str: string whose start chars are to 
            be maximally matched.
        :type subject_str: str
        :param choices: choices of strings to match against
            the starting chars of subject_str
        :type choices: list(str)
        :return: the longest string in choices that overlaps
            with at least part of subject_str. If no overlap,
            returns empty string
        :rtype: str
        '''        
        max_overlap = 0
        result = ""
        for choice in choices:
            # Exact match?
            if choice == subject_str:
                return choice
            if subject_str.startswith(choice) and len(choice) > len(result):
                result = choice
        return result


    #------------------------------------
    # strip_quotes
    #-------------------
    
    @staticmethod
    def strip_quotes(a_str):
        '''
        Given a string, remove any occurrence
        of single or double quotes from the string;
        the returned string will still be a string.
        
        Example:
            strip_quotes("'10'")   ==> '10'
            strip_quotes('"10"')   ==> '10'
            strip_quotes('foobar') ==> 'foobar'

        :param a_str: string from which to snip quotes
        :type a_str: str
        :returns: copy of string with single or double 
            quotes removed
        :rtype: str
        '''
        if type(a_str) != str:
            raise TypeError(f"Arg should be a string, not '{a_str}'")
        return a_str.strip("\"'")

    #------------------------------------
    # clean_stacktrace
    #-------------------
    
    @staticmethod
    def clean_stacktrace(tb, limit=None):
        clean_tb = [frame.replace('\\', '')
                    for frame
                    in tb
                    ]
        if limit is None:
            return clean_tb
        else:
            return clean_tb[-limit:]


    #------------------------------------
    # is_builtin_class
    #-------------------
    
    @staticmethod
    def is_builtin_class(item):
        try:
            if inspect.isclass(item):
                class_obj = item
            else:
                class_obj = item.__class__
            # Eclipse shows an error for 'builtins', but
            # that attr *is* defined:
            return class_obj.__name__ in dir(inspect.builtins)
        except Exception:
            return False
    
    #------------------------------------
    # has_method
    #-------------------
    
    @staticmethod
    def has_method(item, meth_nm):
        '''
        Tests whether item has an attribute
        meth_nm, and whether that attribute is
        a callable. 
        
        :param item: value to be examined
        :type item: Any
        :param meth_nm: name of the method for
            which to test
        :type meth_nm: str
        :return whether or not item has a method
            called meth_nm
        :rtype: bool
        '''
        return hasattr(item, meth_nm) and callable(getattr(item, meth_nm))        
    
        
    #------------------------------------
    # make_toc_fname
    #-------------------

    @staticmethod
    def make_toc_fname(dialog_fname):
        # Use the dialog_fname (the name of the 
        # archive file, and add '_toc' to it:
        
        dialog_nm_p = Path(dialog_fname)
        dialog_file_without_ext = dialog_nm_p.stem
        toc_stem = f"{dialog_file_without_ext}_toc.json"
        toc_fname = str(dialog_nm_p.parent.joinpath(toc_stem))
        return toc_fname

    #------------------------------------
    # construct_filename 
    #-------------------
    
    @staticmethod
    def construct_filename(props_info,
                           prefix=None,
                           suffix=None, 
                           incl_date=False):
        '''
        NOTE: if changes are made to how filenames
              are constructed, check method parse_filename()
              for needed mods
        NOTE: this method ignores all properties that are 
              not in fname_long_2_short. It is therefore specific
              to machine learning contexts. 
              Consider fname_from_props() for more general
              use. 
        Given either:
        
            o a dict of property names and
              associated values, or
            o an object with instance vars 
              named all the long names in
              fname_short_2_long
              
        create a filename that includes all 
        the information held in the keys and 
        values of the dict/obj:

        Ex props_info is a dict:
             {
               'lr' : 0.001,
               'bs' : 32,
               optimizer : 'Adam'
             }
             
        would return the string:
            
            'lr_0.001_bs_32_optimizer_Adam'
            
        Ex props_info is an object:
        
             'net_resnet18_pre_False_lr_0.01_opt_SGD_bs_2_ks_7_folds_3_gray_True_classes_None
        
        I.e. all file elements in fname_short_2_long
        are included in the returned file name.
        
        If obj.num_classes is unavailable, the
        'classes' part of the fname will be 'None'
        
        If a prefix is provided, it will lead the
        string. Example: "Exp" would yield:
        
            'EXP_lr_0.001_bs_32_optimizer_Adam'
            
        If suffix is provided, it will be appended to the
        name: Example, suffix='.csv':
        
            'EXP_lr_0.001_bs_32_optimizer_Adam.csv'
            
        Finally, if incl_date is True, a timestamp is added
        at the start of the returned name, or right after
        the prefix
        
        :param props_info: names and values to include,
            or an object that provides all needed values
            as attributes (instance vars)
        :type props_info: {str : Any}
        :param prefix: leading part of file name
        :type prefix: str
        :param suffix: trailing part of file name
        :type suffix: str
        :param incl_date: whether or not to include current
            data in the file name
        :type incl_date: bool
        :return: a string appropriate for use as a filename
        :rtype: str
        '''
        fname = prefix if prefix is not None else ''
        if incl_date:
            fname += f"_{GeneralUtils.file_timestamp()}"

        if not isinstance(props_info, dict):
            # An obj that promises attrs for each
            # needed value:
            property_dict = GeneralUtils.make_run_props_dict(props_info)
        else:
            property_dict = props_info
            
        for prop_name, prop_val in property_dict.items():
            try:
                short_name = GeneralUtils.fname_long_2_short[prop_name]
            except KeyError:
                # If property is not in the long2short dict,
                # we don't want it as part of the file name:
                continue
            fname += f"_{short_name}_{str(prop_val)}"

        if suffix is not None:
            fname += suffix
            
        return fname

    #------------------------------------
    # parse_filename 
    #-------------------
    
    @staticmethod
    def parse_filename(fname):
        '''
        Given a file name produced by 
        construct_filename(), return a dict
        with the constituent elements and their
        values. The keys are not the abbreviations
        in the filename, but the expanded names
        used in the rest of the code:
        
        Ex:
        From
          pred_2021-03-11T10_59_02_net_resnet18_pretrain_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_gray_True_classes_10.csv
        return
          {net_name   : 'resnet18',
           batch_size : 64
              ...
           }
           
        :param fname: file name to parse
        :type fname: str
        :return: dict with the elements and their values
        :rtype: {str : {int|float|str}}
        '''
        prop_dict = {}
        
        # Remove the file extension:
        fname = str(Path(fname).stem)

        # Get ['lr',0.001,'bs',32,...]:
        fname_els = fname.split('_')
        
        # Find each of the file name elements
        # and their values in the element/val 
        # sequence
        for short_name, long_name in GeneralUtils.fname_short_2_long.items():
            try:
                # Index into the list of fname elements:
                nm_idx = fname_els.index(short_name)
            except ValueError as _e:
                fname_el_val = 'na'
            else:
                # Value of element always follows
                # the element name in filenames:
                val_idx = nm_idx + 1
                try:
                    str_val = fname_els[val_idx]
                    # Convert to proper datatype:
                    fname_el_val = GeneralUtils.fname_el_types[short_name](str_val)
                except IndexError as _e:
                    #raise IndexError(f"Element {short_name} in {fname} has no value for {short_name} ({long_name})")\
                    #    from e
                    fname_el_val = 'na'
                except ValueError:
                    fname_el_val = 'na'
                    
            prop_dict[long_name] = fname_el_val
            
        # Finally: if file name starts with a
        # timestamp, then elements 1,2,3, and 4 
        # comprise the date: 
        #  ['2021-03-11T10','59','02']  ==> '2021-03-11T10_59_02'
        
        match = GeneralUtils.date_at_start_pat.search(fname)
        if match is not None:
            prop_dict['prefix'] = match[1]
            prop_dict['timestamp'] = match[2]
        
        return prop_dict

    #------------------------------------
    # file_timestamp
    #-------------------
    
    @staticmethod
    def file_timestamp():
        '''
        Finds current time, removes milliseconds,
        and replaces colons with underscores. The
        returned string is fit for inclusion in a
        filename.
        
        :return: string for inclusion in filename
        :rtype: str
        '''
        # Remove the msecs part:
        # Replace colons with underscores:
        timestamp = datetime.now().isoformat()
        timestamp = re.sub(r'[.][0-9]{6}', '', timestamp)
        timestamp = timestamp.replace(':', '_')
        return timestamp

    #------------------------------------
    # extract_file_timestamp
    #-------------------
    
    @staticmethod
    def extract_file_timestamp(fname):
        '''
        Given a string---usually a filename---,
        try to find an embedded timestamp formatted
        the way the file_timestamp() method outputs.
        
        Return the timestamp string if found, else None.
        
        :param fname: name to search
        :type fname: str
        :return: the timestamp if found, else None
        :rtype: {None | str}
        '''
        
        # Regex to find substrings created by
        # the file_timestamp() method embedded
        # in a string:
        pat = re.compile(r'.*([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}_[0-9]{2}_[0-9]{2}).*')
        
        the_match = pat.match(fname)
        if the_match is None:
            return None
        
        # Return the extracted timestamp:
        return the_match[1]

    #------------------------------------
    # datetime_from_timestamp
    #-------------------
    
    @staticmethod
    def datetime_from_timestamp(str_timestamp):
        '''
        Given a timestamp that was created by time_for_fname() for use
        in file names, return a datetime.datetime object.
        
        The main task is to turn underscores back into colons.
        
        See also timestamp_from_datetime for the inverse.
        
        :param str_timestamp: timestamp to convert
        :type str_timestamp: str
        :return an equivalent datetime object
        :rtype datetime.datetime
        :raises ValueError if given string is not convertible
        '''
    
        # Already a dt?:
        if isinstance(str_timestamp, datetime):
            return str_timestamp
        dt = datetime.fromisoformat(str_timestamp.replace('_', ':'))
        return dt
                
    #------------------------------------
    # timestamp_from_datetime
    #-------------------
    
    @staticmethod
    def timestamp_from_datetime(dt_timestamp):
        '''
        Given a datetime object, convert it to a date and time
        string that works in file names. I.e. into the format
        produced by time_for_fname()
        
        The main task is to turn colons into colons.
        
        See also datetime_from_timestamp for inverse
        
        :param dt_timestamp: timestamp datetime obj to convert
        :type dt_timestamp: datetime.datetime
        :return an equivalent timestamp string
        :rtype str
        '''
    
        str_timestamp = dt_timestamp.strftime('%Y-%m-%dT%H_%M_%S')
        return str_timestamp


    #------------------------------------
    # seconds_to_hms
    #-------------------
    
    @staticmethod
    def seconds_to_hms(seconds, as_str=True):
        '''
        Given a number of seconds, return either
        a three-tuple (hours, minutes, seconds), 
        or a string hours:minutes:seconds. As string
        is the default.
        
        :param seconds: number of seconds to convert
        :type seconds: int
        :param as_str: whether or not to return a string
            instead of the an hr/min/sec tuple
        :type as_str: bool
        :return the given seconds as 'hr:min:sec', or 
            as (hr, min, sec), depending on the value
            of as_str
        :rtype union(str, tuple)
        '''
        td = timedelta(seconds=seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if as_str:
            return f"{hours}:{minutes}:{seconds}"
        else:
            return hours, minutes, seconds

    #------------------------------------
    # standardize_script_name
    #-------------------

    @staticmethod
    def standardize_script_name(script_path):
        '''
        Given the file path to a script, either as 
        a str or as a pathlib.Path, return the
        script's script_path by which it is known to 
        streamlit. 
        
        Ex: given    /foo/bar/My_UI.py
            returns  'My UI'
        
        :param script_path: path to script whose Streamlit name is sought
        :type script_path: {str | Path}
        :returns name by which the script is known to Streamlit
        :rtype str
        '''
        if type(script_path) == str:
            fpath = Path(script_path)
        elif type(script_path) == pathlib.PosixPath:
            fpath = script_path
        else:
            raise TypeError(f"Script script_path must be str or Path, not {type(script_path)}")
        fname = fpath.stem
        #******return fname.lower().replace("_", " ")
        return fname.replace("_", " ")
    
    #------------------------------------
    # is_json
    #-------------------

    @staticmethod
    def is_json(item):
        try:
            json.loads(item)
            return True
        except Exception:
            return False
        
    #------------------------------------
    # is_email
    #-------------------

    @staticmethod
    def is_email(email_candidate):
        '''
        Tests whether email_candidate is a plausible
        email string. Handles Unicode for foreign emails.
        Does not check domain validity.

        :param email_candidate: item to test
        :type email_candidate: str
        :return: whether or not email_candidate is a
            syntactically valid email
        :rtype: boolean
        '''

        if type(email_candidate) != str:
            return False
        
        # Unicode email pattern
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$|^[\w\.\-\+]+@[\w\-]+\.[a-zA-Z\u0080-\uffff\.]{2,}$'
        
        # Additional checks for maximum length and basic structure
        if len(email_candidate) > 254:  # Maximum length per RFC 5321
            return False
            
        try:
            # Check if the string is valid UTF-8
            email_candidate.encode('utf-8')
            return bool(re.match(pattern, email_candidate))
        except UnicodeError:
            return False
        
    #------------------------------------
    # weighted_choice
    #-------------------            

    @staticmethod
    def weighted_choice(choices, weights, num_samples=1):
        '''
        Given a list of choices, and each choice's weight,
        return num_samples with replacement.

        Weights do not have to add to 1 or 100. They are 
        normalized.

        :param choices: items to pick from
        :type choices: Any
        :param weights: for each choice, a weight that will
            be translated to probabilities
        :type weights: union(int, float)
        :param num_samples: number of returned choices (with replacement)
        :type num_samples: int
        :returns list of choices if num_samples is > 1, else one choice
        :rtype union([Any], Any)
        '''
        choices = random.choices(choices, weights=weights, k=num_samples)
        if num_samples == 1:
            return choices[0]
        else:
            return choices


# -------------------- Context manager for suppressing console output ---------    

#------------------------------------
# suppress_stdout
#-------------------

# This is for use in unittests to keep unittest
# outputs clean:

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout    

        