#!/bin/bash

echo "🔍 Database Status Check"
echo "========================"

# Load environment variables
source /home/azureuser/loldraftai/apps/data-collection/.env

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ Error: DATABASE_URL is not set"
    exit 1
fi

echo "✅ Database URL found"

# Query the database for match counts
echo ""
echo "📊 Match Statistics:"
echo "-------------------"

psql "$DATABASE_URL" -c "
SELECT 
    COUNT(*) as total_matches,
    COUNT(CASE WHEN processed = true THEN 1 END) as processed_matches,
    COUNT(CASE WHEN processed = false THEN 1 END) as unprocessed_matches,
    COUNT(CASE WHEN exported = true THEN 1 END) as exported_matches,
    COUNT(CASE WHEN processed = true AND exported = false THEN 1 END) as ready_for_export
FROM \"Match\";
"

echo ""
echo "📈 Progress Analysis:"
echo "-------------------"

# Get detailed breakdown
TOTAL=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM \"Match\";" | tr -d ' ')
PROCESSED=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM \"Match\" WHERE processed = true;" | tr -d ' ')
UNPROCESSED=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM \"Match\" WHERE processed = false;" | tr -d ' ')
EXPORTED=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM \"Match\" WHERE exported = true;" | tr -d ' ')
READY_FOR_EXPORT=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM \"Match\" WHERE processed = true AND exported = false;" | tr -d ' ')

echo "Total matches: $TOTAL"
echo "Processed matches: $PROCESSED"
echo "Unprocessed matches: $UNPROCESSED"
echo "Exported matches: $EXPORTED"
echo "Ready for export: $READY_FOR_EXPORT"

echo ""
echo "🎯 Export Status:"
echo "----------------"

# Check if we have enough for a full batch (2048 matches)
if [ "$READY_FOR_EXPORT" -ge 2048 ]; then
    echo "✅ Ready for export! ($READY_FOR_EXPORT >= 2048)"
    echo "   extract-to-azure should be uploading data to Azure"
else
    NEEDED=$((2048 - READY_FOR_EXPORT))
    echo "⏳ Waiting for more processed matches..."
    echo "   Need $NEEDED more processed matches for full batch (2048)"
fi

echo ""
echo "📊 Processing Rate Estimate:"
echo "---------------------------"

# Calculate percentage processed
if [ "$TOTAL" -gt 0 ]; then
    PERCENT_PROCESSED=$((PROCESSED * 100 / TOTAL))
    echo "Processing progress: $PERCENT_PROCESSED% ($PROCESSED/$TOTAL)"
else
    echo "No matches found in database"
fi

echo ""
echo "🔍 Service Status:"
echo "-----------------"
pm2 list | grep -E "(processMatches|extract-to-azure)"

echo ""
echo "✅ Database check complete!" 