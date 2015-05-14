from distutils.core import setup

setup(name='htrc-acs',
      description='A tool for querying the HTRC based on LoC metadata',
      author='HTRC-InPhO Advanced Collaborative Support',
      author_email='inpho@indiana.edu',
      url='http://github.com/inpho/htrc-acs',
      license='MIT',
      requires=['python-skos==0.0.4.3',
                'RDFLib==2.4.2',
                'iso8601plus'])
