https://towardsdatascience.com/beginners-guide-to-custom-environments-in-openai-s-gym-989371673952

To install our environments, type the following command in power shell 
python -m pip install -e our-environments

When adding new environments, please do not forget add the following under def __int__():
self.name=os.path.splitext(os.path.basename(__file__))[0]