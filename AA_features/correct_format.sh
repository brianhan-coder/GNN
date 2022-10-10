file=AA_CoilParameter.dat
echo $file
sed -i '' '/^$/d' $file #remove empty line
sed -i '' 's/://g' $file
sed -i '' 's/  / /g' $file
sed -i '' 's/ *$//' $file
awk '{print toupper($0)}' $file > tmp3 && mv tmp3 $file
#rm tmp*