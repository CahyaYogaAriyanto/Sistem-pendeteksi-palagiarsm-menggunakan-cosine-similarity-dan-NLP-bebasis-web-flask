import pdfkit

config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

html = "<h1>Hello, PDF!</h1><p>Ini adalah percobaan.</p>"

pdfkit.from_string(html, "hasil_test.pdf", configuration=config)