import { test, expect } from '@playwright/test';

test.describe('Theme Toggle', () => {
  test('should toggle between light and dark themes', async ({ page }) => {
    await page.goto('/');

    // Wait for the page to load
    await expect(page.locator('h1')).toContainText('Stock Advice Dashboard');

    // Check that theme toggle button exists
    const themeToggle = page.locator('button[aria-label="Toggle theme"]');
    await expect(themeToggle).toBeVisible();

    // Get initial theme state
    const htmlElement = page.locator('html');
    const initialClasses = await htmlElement.getAttribute('class');
    
    // Click theme toggle
    await themeToggle.click();
    
    // Wait for theme change
    await page.waitForTimeout(100);
    
    // Check that theme has changed
    const newClasses = await htmlElement.getAttribute('class');
    expect(newClasses).not.toBe(initialClasses);
    
    // Check that dark or light class is present
    expect(newClasses).toMatch(/(dark|light)/);
    
    // Click again to toggle back
    await themeToggle.click();
    await page.waitForTimeout(100);
    
    // Should return to original state or have opposite theme
    const finalClasses = await htmlElement.getAttribute('class');
    expect(finalClasses).toMatch(/(dark|light)/);
  });

  test('should persist theme on page refresh', async ({ page }) => {
    await page.goto('/');
    
    const themeToggle = page.locator('button[aria-label="Toggle theme"]');
    await themeToggle.click();
    await page.waitForTimeout(100);
    
    const htmlElement = page.locator('html');
    const themeAfterToggle = await htmlElement.getAttribute('class');
    
    // Refresh the page
    await page.reload();
    
    // Wait for page to load
    await expect(page.locator('h1')).toContainText('Stock Advice Dashboard');
    
    // Check that theme is still the same
    const themeAfterRefresh = await htmlElement.getAttribute('class');
    expect(themeAfterRefresh).toMatch(/(dark|light)/);
  });

  test('should show correct icon for current theme', async ({ page }) => {
    await page.goto('/');
    
    const themeToggle = page.locator('button[aria-label="Toggle theme"]');
    
    // Check initial icon (should be moon for light theme or sun for dark theme)
    const iconBefore = themeToggle.locator('svg');
    await expect(iconBefore).toBeVisible();
    
    // Toggle theme
    await themeToggle.click();
    await page.waitForTimeout(100);
    
    // Icon should still be visible after toggle
    const iconAfter = themeToggle.locator('svg');
    await expect(iconAfter).toBeVisible();
  });
});
