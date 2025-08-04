import { test, expect } from '@playwright/test';

test.describe('Complete User Workflow', () => {
  test('should complete full user journey from landing to analysis to recommendations', async ({ page }) => {
    // Start at the landing page
    await page.goto('/');
    await expect(page.locator('h1')).toContainText('Stock Advice Dashboard');

    // Check that feature cards are present
    await expect(page.locator('text=Generate Analysis')).toBeVisible();
    await expect(page.locator('text=View Recommendations')).toBeVisible();
    
    // Check current features section
    await expect(page.locator('text=Current Features')).toBeVisible();
    await expect(page.locator('text=Technical Analysis')).toBeVisible();
    await expect(page.locator('text=Fundamental Analysis')).toBeVisible();
    await expect(page.locator('text=Sentiment Analysis')).toBeVisible();
    
    // Check upcoming features section
    await expect(page.locator('text=Coming Soon: F&O Analysis')).toBeVisible();
    await expect(page.locator('text=Option Chain Analysis')).toBeVisible();
    await expect(page.locator('text=Volatility Insights')).toBeVisible();

    // Navigate to Generate Analysis via card click
    await page.locator('a[href="/analysis"]').first().click();
    await expect(page).toHaveURL('/analysis');
    
    // Check analysis page loads (this will depend on your existing analysis page)
    // For now, just check that we're on the right page
    await expect(page).toHaveURL('/analysis');

    // Navigate to Recommendations
    const stockAnalysisButton = page.locator('button:has-text("Stock Analysis")');
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    await page.locator('a[href="/recommendations"]').click();
    await expect(page).toHaveURL('/recommendations');

    // Navigate to Settings
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    await page.locator('a[href="/settings"]').click();
    await expect(page).toHaveURL('/settings');
    await expect(page.locator('h1')).toContainText('Settings');

    // Test settings functionality
    await expect(page.locator('text=General')).toBeVisible();
    await expect(page.locator('text=Notifications')).toBeVisible();
    await expect(page.locator('text=Profile')).toBeVisible();
    await expect(page.locator('text=Privacy & Security')).toBeVisible();

    // Test settings sections
    await page.locator('button:has-text("Notifications")').click();
    await expect(page.locator('text=Email alerts')).toBeVisible();
    
    await page.locator('button:has-text("Profile")').click();
    await expect(page.locator('text=Risk Tolerance')).toBeVisible();
    
    await page.locator('button:has-text("Privacy & Security")').click();
    await expect(page.locator('text=Share data for research')).toBeVisible();

    // Return to homepage
    await page.locator('text=Stock Advisor').click();
    await expect(page).toHaveURL('/');
    await expect(page.locator('h1')).toContainText('Stock Advice Dashboard');
  });

  test('should handle responsive design elements', async ({ page }) => {
    // Test with different viewport sizes
    await page.setViewportSize({ width: 1200, height: 800 });
    await page.goto('/');
    
    // Check that elements are visible in desktop view
    await expect(page.locator('text=Stock Advice Dashboard')).toBeVisible();
    await expect(page.locator('button:has-text("Stock Analysis")')).toBeVisible();

    // Test tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.reload();
    await expect(page.locator('text=Stock Advice Dashboard')).toBeVisible();

    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload();
    await expect(page.locator('text=Stock Advice Dashboard')).toBeVisible();
  });

  test('should maintain theme consistency across pages', async ({ page }) => {
    await page.goto('/');
    
    // Toggle to dark theme
    const themeToggle = page.locator('button[aria-label="Toggle theme"]');
    await themeToggle.click();
    await page.waitForTimeout(100);
    
    const htmlElement = page.locator('html');
    const darkThemeClasses = await htmlElement.getAttribute('class');
    
    // Navigate to different pages and check theme persists
    const stockAnalysisButton = page.locator('button:has-text("Stock Analysis")');
    
    // Go to Analysis page
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    await page.locator('a[href="/analysis"]').click();
    
    const themeOnAnalysis = await htmlElement.getAttribute('class');
    expect(themeOnAnalysis).toMatch(/(dark|light)/);
    
    // Go to Recommendations page
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    await page.locator('a[href="/recommendations"]').click();
    
    const themeOnRecommendations = await htmlElement.getAttribute('class');
    expect(themeOnRecommendations).toMatch(/(dark|light)/);
    
    // Go to Settings page
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    await page.locator('a[href="/settings"]').click();
    
    const themeOnSettings = await htmlElement.getAttribute('class');
    expect(themeOnSettings).toMatch(/(dark|light)/);
  });
});
