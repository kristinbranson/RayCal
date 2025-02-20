class Employee():
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = None
        email = self.email
        print(email)
        #self.email = first + '.' + last + '@company.com'
    
    @property
    def email(self):
        return  'temp@company.com'
    
    @email.setter
    def email(self, value):
        pass
    #    self.first, self.last = value.split('@')[0].split('.')


    def send_email(self):
        return 'Email sent to {}'.format(self.email)


    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
class Developer(Employee):
    def __init__(self, first, last, pay, prog_lang):
        super().__init__(first, last, pay)
        self.prog_lang = prog_lang

if __name__ == '__main__':
    developer = Developer('Aniket', 'Ravan', 50000, 'Python')
    