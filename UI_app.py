from tkinter import *
from math import *
from fetch_answer import *

#https://stackoverflow.com/a/55943748
main = Tk()
main.title('Factoid-QA System')
def tfidf():
    blank.delete(0, END)
    qu = str(num1.get())
    ans_lists = get_doc(qu)
    Ans = get_answer(qu, ans_lists)
    blank.insert(0, Ans)
def w2v():
    blank.delete(0, END)
    qu = str(num1.get())
    ans_lists = get_doc(qu)
    Ans = get_ans_w2v(qu, ans_lists)
    blank.insert(0, Ans)
def clear():
    blank.delete(0, END)
    num2.delete(0, END)
    num1.delete(0, END)


main.geometry('500x500')
Label(main, text = "Enter Your Qestion:").grid(row=0)
Label(main, text = "The Answer is:").grid(row=1)


num1 = Entry(main)
blank = Entry(main)


num1.grid(row=0, column=1)
blank.grid(row=1, column=1)



Button(main, text='Quit', command=main.destroy).grid(row=4, column=0, sticky=W)
Button(main, text='Tf-Idf', command=tfidf).grid(row=0, column=3, sticky=W,)
Button(main, text='w2v', command=w2v).grid(row=0, column=4, sticky=W)
Button(main, text='Clear', command=clear).grid(row=0, column=5, sticky=W)

mainloop()

