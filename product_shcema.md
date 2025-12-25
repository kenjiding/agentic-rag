import { pgTable, serial, varchar, decimal, integer, text, jsonb, timestamp, boolean } from 'drizzle-orm/pg-core'
import { relations } from 'drizzle-orm'

export const brands = pgTable('brands', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 100 }).notNull().unique(),
  image: varchar('image', { length: 255 }).notNull().unique()
})

export const mainCategories = pgTable('main_categories', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 100 }).notNull(),
  brandId: integer('brand_id').notNull().references(() => brands.id)
})

export const subCategories = pgTable('sub_categories', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 100 }).notNull(),
  mainCategoryId: integer('main_category_id').notNull().references(() => mainCategories.id)
})

export const products = pgTable('products', {
  id: serial('id').primaryKey(),
  name: varchar('name', { length: 200 }).notNull(),
  modelNumber: varchar('model_number', { length: 100 }),
  subCategoryId: integer('sub_category_id').references(() => subCategories.id),
  mainCategoryId: integer('main_category_id').references(() => mainCategories.id),
  brandId: integer('brand_id').notNull().references(() => brands.id),
  description: text('description'),
  features: text('features'),
  price: decimal('price', { precision: 10, scale: 2 }),
  // stock: decimal('stock'),
  images: jsonb('images'),
  special: boolean('special').default(false),
  createdAt: timestamp('created_at').defaultNow(),
  updatedAt: timestamp('updated_at').defaultNow()
})

// 关系定义
export const brandsRelations = relations(brands, ({ many }) => ({
  mainCategories: many(mainCategories),
  products: many(products)
}))

export const mainCategoriesRelations = relations(mainCategories, ({ one, many }) => ({
  brand: one(brands, {
    fields: [mainCategories.brandId],
    references: [brands.id]
  }),
  subCategories: many(subCategories)
}))

export const subCategoriesRelations = relations(subCategories, ({ one, many }) => ({
  mainCategory: one(mainCategories, {
    fields: [subCategories.mainCategoryId],
    references: [mainCategories.id]
  }),
  products: many(products)
}))

export const productsRelations = relations(products, ({ one }) => ({
  brand: one(brands, {
    fields: [products.brandId],
    references: [brands.id]
  }),
  subCategory: one(subCategories, {
    fields: [products.subCategoryId],
    references: [subCategories.id]
  })
}))

// 定义类型
export type Brand = typeof brands.$inferSelect
export type NewBrand = typeof brands.$inferInsert

export type MainCategory = typeof mainCategories.$inferSelect
export type NewMainCategory = typeof mainCategories.$inferInsert

export type SubCategory = typeof subCategories.$inferSelect
export type NewSubCategory = typeof subCategories.$inferInsert

export type Product = typeof products.$inferSelect
export type NewProduct = typeof products.$inferInsert