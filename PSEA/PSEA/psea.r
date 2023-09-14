rm(list = ls())

## import library
library(PSEA)
library(readxl)
library(openxlsx)

## read dataset
reference_expression   = read.csv2("prot_use.csv",encoding = "uft-8",sep = ";",row.names = 1)
reference_expression_n = apply(reference_expression,2,as.numeric)
gene_id                = read.csv2("prot_use.csv", encoding = "utf-8", sep = ";")
gene_id                = gene_id[,1]

row_names <- list()
for ( n in 1:4891){
  app <- list(gene_id[n])
  row_names <- c(row_names,app)
  }
rownames(reference_expression_n) <- row_names


## build the expression list
## The element inside c is considered to be the expression related to the same pathway.
sheet1_probesets <- list("VNN1","CYP7B1","HLA-DQB1","ABCC1","ITPR1","PEX3")
sheet1_reference <- marker(reference_expression_n,sheet1_probesets)
print(sheet1_reference)

sheet2_probesets <- list("RIPK2", "TFPI","SLC17A5","RIPK2")
sheet2_reference <- marker(reference_expression_n,sheet2_probesets)
print(sheet2_reference)

sheet3_probesets <- list("HAL","ACY3","UROC1","GLS2",c("CYP4A11","CYP4A22"),"NMNAT1","SDS","ACMSD","FES",
                        c("HAL","ACY3","UROC1"),
                        c("HAL","GLS2"),
                        c("ACY3","GLS2"))
sheet3_reference <- marker(reference_expression_n,sheet3_probesets)
print(sheet3_reference)

sheet4_probesets <- list("CYP7A1","CA14","AHCYL2","ADSS1","CDA","PRODH","TRIP6","FBXO2")
sheet4_reference <- marker(reference_expression_n,sheet4_probesets)
print(sheet4_reference)

sheet5_probesets <- list(c("CYP2B6","FMO4","UGT1A3"),
                         c("PHKG2","GYS2","BAD"),
                         c("UGT1A3","GYS2"),
                         c("CYP2B6","UGT1A3"),
                         c("ACADL","CYP8B1"),
                         c("CYP2B6","UGT1A3"),
                         c("PEX14","HAO2"),
                         "CYP2B6","FMO4","UGT1A3","PHKG2","GYS2","BAD","ACADL","CYP8B1","PEX14","HAO2",
                         "DDC","AGXT2","ACSM5","PFKFB1","ACOT12","ABCB11")
sheet5_reference <- marker(reference_expression_n,sheet5_probesets)
print(sheet5_reference)

sheet6_probesets <- list("CHST14","CXCL12","PPP1R14A")
sheet6_reference <- marker(reference_expression_n,sheet6_probesets)
print(sheet6_reference)

sheet7_probesets <- list("PANK1","NAGLU","HNF1B","RELN","UBR5")
sheet7_reference <- marker(reference_expression_n,sheet7_probesets)
print(sheet7_reference)

sheet8_probesets <- list("RNASE3","PRKCB","ALOX5","PLTP","ELANE")
sheet8_reference <- marker(reference_expression_n,sheet8_probesets)
print(sheet8_reference)

sheet9_probesets <- list("XPC","THBS4","MAGI1")
sheet9_reference <- marker(reference_expression_n,sheet9_probesets)
print(sheet9_reference)

sheet10_probesets <- list(c("WAS","DOCK2"),"WAS","ITGA3","ZAP70","TCIRG1","SIPA1","AMPD3")
sheet10_reference <- marker(reference_expression_n,sheet10_probesets)
print(sheet10_reference)

# import group
gene_group      = read.csv2("group.csv", encoding = "utf-8", sep = ";", row.names = 1)

x <- data.frame(sheet1_reference,
                sheet2_reference,
                sheet3_reference,
                sheet4_reference,
                sheet5_reference,
                sheet6_reference,
                sheet7_reference,
                sheet8_reference,
                sheet9_reference,
                sheet10_reference,
                gene_group)

wb <- createWorkbook()
addWorksheet(wb, "sheet1")
writeData(
  wb,# workbook对象
  "sheet1",# sheetindex或者名字
  x,# 数据
  colNames = TRUE,# 以下忽略
  rowNames = TRUE
)
saveWorkbook(wb, file = "data.xlsx", overwrite = TRUE)
