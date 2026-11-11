"""Check exported Excel controls and blank human fields, without altering the workbook."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

folder=Path(__file__).resolve().parent.parent
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
for reviewer in ['A','B']:
    file=folder/f'M1独立人审-{reviewer}.xlsx'
    with zipfile.ZipFile(file) as z:
        sheet=ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        pane=sheet.find('s:sheetViews/s:sheetView/s:pane',ns)
        validations=sheet.findall('s:dataValidations/s:dataValidation',ns)
        ranges=[v.attrib['sqref'] for v in validations]
        assert 'F10:L65' in ranges and 'G3' in ranges,ranges
        for column in 'FGHIJKL':
            for row in range(10,66):
                cell=sheet.find(f'.//s:c[@r="{column}{row}"]',ns)
                if cell is not None:
                    assert cell.find('s:f',ns) is None
                    assert cell.find('s:v',ns) is None
                    assert not ''.join(cell.itertext()).strip()
        assert pane is not None
        assert pane.attrib.get('xSplit')=='2' and pane.attrib.get('ySplit')=='9',pane.attrib
        table=ET.fromstring(z.read('xl/tables/table1.xml'))
        assert table.attrib['ref']=='A9:O65'
        assert table.find('s:autoFilter',ns) is not None
        assert len(table.findall('s:tableColumns/s:tableColumn',ns))==15
        print(json.dumps(dict(reviewer=reviewer,file=file.name,blank_human_fields=392,
            freeze=pane.attrib,filter_range=table.attrib['ref'],validation_ranges=ranges),ensure_ascii=False))
