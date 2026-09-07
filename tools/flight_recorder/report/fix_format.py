from pathlib import Path
import os
import json, re, hashlib
from docx import Document
from docx.shared import Inches, Pt, Twips, RGBColor
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.enum.table import WD_TABLE_ALIGNMENT
R=Path(os.environ.get("FLIGHT_REPORT_DIR", ".")).resolve()
src=R/"logits-design.docx"
d=Document(src)
GEOMETRY={'page_width': 12240, 'page_height': 15840, 'left_margin': 1037, 'right_margin': 1037, 'top_margin': 979, 'bottom_margin': 979}
TYPE_SIZES={'Normal': 11.0, 'Title': 24.0, 'Heading 1': 17.0, 'Heading 2': 14.5}
def content(doc):
 return [p.text.replace('\u200b','') for p in doc.paragraphs]+[c.text.replace('\u200b','') for t in doc.tables for row in t.rows for c in row.cells]
original=content(d)
for sec in d.sections:
 for prop in ["page_width","page_height","left_margin","right_margin","top_margin","bottom_margin"]:
  setattr(sec,prop,Twips(GEOMETRY[prop]))
# Adopt the reference's type hierarchy; retain the installed CJK font.
for n in ["Normal","Title","Heading 1","Heading 2"]:
 a=d.styles[n]
 a.font.size=Pt(TYPE_SIZES[n])
 if n!="Normal":a.font.color.rgb=RGBColor(0,0,0)
# Preserve original paragraph rhythm to avoid needless pagination changes.
W=10080
spec=[]
def child(parent,tag):
 el=parent.find(qn("w:"+tag))
 if el is None:el=OxmlElement("w:"+tag);parent.append(el)
 return el
for ti,t in enumerate(d.tables):
 old=[c.width.twips for c in t.columns]
 widths=[round(x*W/sum(old)) for x in old];widths[-1]=W-sum(widths[:-1])
 t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
 pr=t._tbl.tblPr
 child(pr,"tblW").attrib.update({qn("w:type"):"dxa",qn("w:w"):str(W)})
 child(pr,"tblInd").attrib.update({qn("w:type"):"dxa",qn("w:w"):"0"})
 child(pr,"tblCellSpacing").attrib.update({qn("w:type"):"dxa",qn("w:w"):"0"})
 for el in list(pr.findall(qn("w:tblBorders"))):pr.remove(el)
 pr.append(parse_xml('<w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" xmlns:mo="http://schemas.microsoft.com/office/mac/office/2008/main" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" xmlns:mv="urn:schemas-microsoft-com:mac:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" xmlns:v="urn:schemas-microsoft-com:vml" xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:w10="urn:schemas-microsoft-com:office:word" xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape"><w:top w:val="single" w:sz="8" w:color="D9D9D9"/><w:left w:val="single" w:sz="8" w:color="D9D9D9"/><w:bottom w:val="single" w:sz="8" w:color="D9D9D9"/><w:right w:val="single" w:sz="8" w:color="D9D9D9"/><w:insideH w:val="single" w:sz="8" w:color="D9D9D9"/><w:insideV w:val="single" w:sz="8" w:color="D9D9D9"/></w:tblBorders>'))
 for col,w in zip(t.columns,widths):col.width=Twips(w)
 for ri,row in enumerate(t.rows):
  tr=row._tr.get_or_add_trPr()
  child(tr,"cantSplit")
  if ri==0:child(tr,"tblHeader")
  for ci,c in enumerate(row.cells):
   c.width=Twips(widths[ci])
   cp=c._tc.get_or_add_tcPr()
   nowrap=cp.find(qn("w:noWrap"))
   if nowrap is not None:cp.remove(nowrap)
   sh=child(cp,"shd");sh.set(qn("w:fill"),"F2F2F2" if ri==0 else "FFFFFF")
   for pa in c.paragraphs:
    pa.paragraph_format.space_before=Pt(0);pa.paragraph_format.space_after=Pt(2)
    pa.paragraph_format.line_spacing=1.12
    pa.paragraph_format.keep_with_next=ri==0
    pa.paragraph_format.widow_control=True
    for run in pa.runs:
     # Provide real wrap opportunities within long source paths; tokens unchanged.
     run.text=re.sub(r'([A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_.-]*',
       lambda m:m.group(0).replace('/','/\u200b').replace('_','_\u200b'),run.text)
     run.font.size=Pt(9);run.font.name="Noto Sans CJK SC"
     run._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"),"Noto Sans CJK SC")
 spec.append({"table":ti,"width_twips":W,"columns_twips":widths})
assert content(d)==original
for im in d.inline_shapes:
 assert im.width<=d.sections[0].page_width-d.sections[0].left_margin-d.sections[0].right_margin
out=R/"logits-design-formatted.docx";d.save(out)
check=Document(out)
for t in check.tables:
 grid=[x.width.twips for x in t.columns]
 assert sum(grid)==W
 assert t._tbl.tblPr.find(qn('w:tblW')).get(qn('w:w'))==str(W)
 for row in t.rows:
  assert [c.width.twips for c in row.cells]==grid
assert content(check)==original
info={"output":str(out),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"bytes":out.stat().st_size,"tables":spec,"body_width_twips":round((check.sections[0].page_width-check.sections[0].left_margin-check.sections[0].right_margin)/635),"content_unchanged":True}
(R/"format-check.json").write_text(json.dumps(info,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in info.items() if k!="tables"},ensure_ascii=False))
