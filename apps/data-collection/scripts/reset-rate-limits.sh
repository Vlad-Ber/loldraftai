#!/bin/bash

# Script to reset rate limits by stopping services temporarily
# This helps clear any accumulated rate limit issues

echo "🔄 Resetting rate limits..."

echo "⏸️  Stopping all services..."
pm2 stop all

echo "⏳ Waiting 2 minutes to clear any rate limit issues..."
sleep 120

echo "▶️  Starting services with extremely conservative rate limiting..."
pm2 start all

echo "✅ Services restarted with conservative rate limiting"
echo ""
echo "📊 New Rate Limits:"
echo "   - collectMatchIds: 0.067 requests/second per service"
echo "   - processMatches: 0.083 requests/second per service"
echo "   - updateLadder: 0.05 requests/second per service"
echo "   - Total: ~0.2 requests/second across all services"
echo ""
echo "🔍 Monitor with: pm2 logs" 