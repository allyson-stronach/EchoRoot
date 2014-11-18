import re 
#regular expressions operate on strings
#read about the regex backslash plague here: https://docs.python.org/2/howto/regex.html
	#=> be sure to convert all regular python strings with raw strings:
	#"ab*" becomes r"ab*"


#compile() accepts an optional flags argument, like p = re.compile('ab*', re.IGNORECASE)
p = re.compile('[a-z]+')
p  #doctest: +ELLIPSIS

p.match("")
print "this shouldn't match our string, expect NONE" p.match("")

m = p.match('tempo')
m

