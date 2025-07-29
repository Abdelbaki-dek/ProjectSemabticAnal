import io
import pandas as pd
from fpdf import FPDF
import matplotlib.pyplot as plt

def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Analyse')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def to_pdf(df, themes, sous_themes):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "Rapport d'analyse sémantique", ln=1, align='C')

    pdf.cell(0, 10, "Thèmes détectés :", ln=1)
    for theme in themes:
        pdf.cell(0, 8, f"- {theme}", ln=1)

    pdf.cell(0, 10, "Sous-thèmes :", ln=1)
    for stheme_list in sous_themes:
        for stheme in stheme_list:
            pdf.cell(0, 8, f"  * {stheme}", ln=1)

    pdf.cell(0, 10, "Aperçu des commentaires :", ln=1)
    for idx, row in df.head(10).iterrows():
        comment = row.get('commentaire', '')[:100].replace('\n', ' ')
        pdf.multi_cell(0, 8, f"- {comment}...")

    # TODO : ajouter graphiques images générés + autres détails si besoin

    pdf_output = pdf.output(dest='S').encode('latin-1')
    return pdf_output

