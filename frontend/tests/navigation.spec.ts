import { test, expect } from '@playwright/test';

test.describe('Navigation', () => {
  test('should navigate to all main pages', async ({ page }) => {
    await page.goto('/');

    // Check homepage loads
    await expect(page.locator('h1')).toContainText('Stock Advice Dashboard');

    // Test Stock Analysis dropdown
    const stockAnalysisButton = page.locator('button:has-text("Stock Analysis")');
    await expect(stockAnalysisButton).toBeVisible();
    
    // Click to open dropdown
    await stockAnalysisButton.click();
    
    // Wait for dropdown to appear
    await page.waitForTimeout(100);
    
    // Check dropdown items are visible
    await expect(page.locator('text=Generate Analysis')).toBeVisible();
    await expect(page.locator('text=View Recommendations')).toBeVisible();
    await expect(page.locator('text=Settings')).toBeVisible();
    
    // Navigate to Generate Analysis
    await page.locator('a[href="/analysis"]').click();
    await expect(page).toHaveURL('/analysis');
    
    // Go back to home and test Recommendations
    await page.goto('/');
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    await page.locator('a[href="/recommendations"]').click();
    await expect(page).toHaveURL('/recommendations');
    
    // Go back to home and test Settings
    await page.goto('/');
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    await page.locator('a[href="/settings"]').click();
    await expect(page).toHaveURL('/settings');
    await expect(page.locator('h1')).toContainText('Settings');
  });

  test('should highlight active menu item', async ({ page }) => {
    // Test Analysis page
    await page.goto('/analysis');
    
    const stockAnalysisButton = page.locator('button:has-text("Stock Analysis")');
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    
    // Check that Generate Analysis is highlighted as active
    const analysisLink = page.locator('a[href="/analysis"]');
    const analysisClasses = await analysisLink.getAttribute('class');
    expect(analysisClasses).toContain('bg-blue-50');
    
    // Test Recommendations page
    await page.goto('/recommendations');
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    
    const recommendationsLink = page.locator('a[href="/recommendations"]');
    const recommendationsClasses = await recommendationsLink.getAttribute('class');
    expect(recommendationsClasses).toContain('bg-blue-50');
    
    // Test Settings page
    await page.goto('/settings');
    await stockAnalysisButton.click();
    await page.waitForTimeout(100);
    
    const settingsLink = page.locator('a[href="/settings"]');
    const settingsClasses = await settingsLink.getAttribute('class');
    expect(settingsClasses).toContain('bg-blue-50');
  });

  test('should close dropdown when clicking outside', async ({ page }) => {
    await page.goto('/');
    
    const stockAnalysisButton = page.locator('button:has-text("Stock Analysis")');
    await stockAnalysisButton.click();
    
    // Wait for dropdown to appear
    await page.waitForTimeout(100);
    await expect(page.locator('text=Generate Analysis')).toBeVisible();
    
    // Click outside the dropdown (on the logo)
    await page.locator('text=Stock Advisor').click();
    
    // Wait for dropdown to close
    await page.waitForTimeout(200);
    
    // Dropdown should be hidden
    await expect(page.locator('text=Generate Analysis')).not.toBeVisible();
  });

  test('should show logo and return to home when clicked', async ({ page }) => {
    await page.goto('/analysis');
    
    // Check logo is visible
    const logo = page.locator('text=Stock Advisor');
    await expect(logo).toBeVisible();
    
    // Click logo to return home
    await logo.click();
    await expect(page).toHaveURL('/');
    await expect(page.locator('h1')).toContainText('Stock Advice Dashboard');
  });
});
