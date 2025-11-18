"""
Marketplace Plugin

Adds marketplace functionality for buying/selling items within communities
"""
from typing import Dict, Any
from plugins.base import PluginBase, PluginType, HookType, PluginContext
import logging

logger = logging.getLogger(__name__)


class MarketplacePlugin(PluginBase):
    """
    Marketplace plugin - buy/sell items within communities

    Configuration:
        - payment_gateway: Payment integration (stripe, paypal) (default: stripe)
        - commission_rate: Platform commission percentage (default: 5.0)
        - require_verification: Require seller verification (default: True)
        - escrow_enabled: Use escrow for transactions (default: True)
        - max_price: Maximum item price (default: unlimited)
    """

    @property
    def name(self) -> str:
        return "Marketplace"

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def description(self) -> str:
        return "Buy and sell items within communities with integrated payments and seller ratings"

    @property
    def plugin_type(self) -> PluginType:
        return PluginType.CONTENT

    async def initialize(self) -> bool:
        """Initialize the marketplace plugin"""
        logger.info(f"Initializing {self.name} plugin v{self.version}")

        # Register hooks
        self.register_hook(HookType.BEFORE_POST_CREATE, self._validate_listing)
        self.register_hook(HookType.AFTER_POST_CREATE, self._create_listing)
        self.register_hook(HookType.POST_RENDER, self._render_listing)
        self.register_hook(HookType.USER_REPUTATION_CHANGE, self._update_seller_rating)

        # Configuration
        self.payment_gateway = self.get_config("payment_gateway", "stripe")
        self.commission_rate = self.get_config("commission_rate", 5.0)
        self.require_verification = self.get_config("require_verification", True)
        self.escrow_enabled = self.get_config("escrow_enabled", True)
        self.max_price = self.get_config("max_price", None)

        logger.info(
            f"{self.name} plugin initialized with {self.payment_gateway} gateway"
        )
        return True

    async def shutdown(self) -> bool:
        """Shutdown the marketplace plugin"""
        logger.info(f"Shutting down {self.name} plugin")
        return True

    async def _validate_listing(self, context: PluginContext) -> PluginContext:
        """Validate marketplace listing before post creation"""
        post_type = context.get("post_type")

        if post_type != "marketplace":
            return context

        listing_data = context.get("listing_data")

        if not listing_data:
            context.set("error", "Listing data is required for marketplace posts")
            return context

        # Validate listing structure
        required_fields = ["title", "price", "condition", "category"]
        for field in required_fields:
            if field not in listing_data:
                context.set("error", f"Listing {field} is required")
                return context

        # Validate price
        price = listing_data.get("price", 0)
        if price <= 0:
            context.set("error", "Price must be greater than 0")
            return context

        if self.max_price and price > self.max_price:
            context.set("error", f"Price cannot exceed ${self.max_price}")
            return context

        # Check seller verification if required
        user_id = context.get("user_id")
        if self.require_verification:
            is_verified = self._check_seller_verification(user_id)
            if not is_verified:
                context.set("error", "Seller verification required to create listings")
                return context

        # Validation passed
        context.set("listing_validated", True)
        logger.debug(f"Listing validated: {listing_data.get('title')} - ${price}")

        return context

    async def _create_listing(self, context: PluginContext) -> PluginContext:
        """
        Create marketplace listing after post creation

        Hook: AFTER_POST_CREATE
        """
        post_type = context.get("post_type")

        if post_type != "marketplace":
            return context

        post_id = context.get("post_id")
        listing_data = context.get("listing_data")

        if not listing_data:
            return context

        # In a real implementation, this would:
        # 1. Create listing record in database
        # 2. Set up payment integration (Stripe Connect, etc.)
        # 3. Initialize escrow if enabled
        # 4. Create seller dashboard entry

        logger.info(
            f"Created marketplace listing for post {post_id}: "
            f"{listing_data.get('title')} - ${listing_data.get('price')}"
        )

        # Store listing metadata in context
        context.set("listing_created", True)
        context.set("listing_id", f"listing_{post_id}")

        return context

    async def _render_listing(self, context: PluginContext) -> PluginContext:
        """
        Render marketplace listing UI in post

        Hook: POST_RENDER
        """
        post_type = context.get("post_type")

        if post_type != "marketplace":
            return context

        listing_data = context.get("listing_data")
        user_id = context.get("user_id")

        if not listing_data:
            return context

        # Check if user is the seller
        seller_id = listing_data.get("seller_id")
        is_seller = (user_id == seller_id)

        # Build listing HTML
        html = self._build_listing_html(
            listing_data,
            is_seller=is_seller,
        )

        context.set("listing_html", html)
        logger.debug(f"Rendered marketplace listing for user {user_id}")

        return context

    async def _update_seller_rating(self, context: PluginContext) -> PluginContext:
        """
        Update seller rating when reputation changes

        Hook: USER_REPUTATION_CHANGE
        """
        user_id = context.get("user_id")
        reputation_change = context.get("reputation_change", 0)

        # Update seller rating based on reputation
        # TODO: Implement seller rating calculation

        logger.debug(
            f"Updated seller rating for user {user_id}: {reputation_change:+d}"
        )

        return context

    def _check_seller_verification(self, user_id: str) -> bool:
        """Check if seller is verified"""
        # Placeholder - would query database
        # Check for ID verification, payment method, etc.
        return True

    def _build_listing_html(
        self, listing_data: Dict[str, Any], is_seller: bool
    ) -> str:
        """Build marketplace listing HTML markup"""
        title = listing_data.get("title", "")
        price = listing_data.get("price", 0)
        condition = listing_data.get("condition", "")
        category = listing_data.get("category", "")
        description = listing_data.get("description", "")
        images = listing_data.get("images", [])
        seller_name = listing_data.get("seller_name", "")
        seller_rating = listing_data.get("seller_rating", 0)
        status = listing_data.get("status", "available")

        html = f'<div class="marketplace-listing">'

        # Images gallery
        if images:
            html += f'<div class="listing-images">'
            html += f'<img src="{images[0]}" alt="{title}" class="listing-main-image">'
            if len(images) > 1:
                html += f'<div class="listing-thumbnails">'
                for img in images[1:4]:  # Show up to 3 thumbnails
                    html += f'<img src="{img}" alt="{title}">'
                html += f'</div>'
            html += f'</div>'

        # Listing details
        html += f'<div class="listing-details">'
        html += f'<h2 class="listing-title">{title}</h2>'

        html += f'<div class="listing-price">${price:.2f}</div>'

        html += f'<div class="listing-meta">'
        html += f'<span class="listing-condition">{condition}</span>'
        html += f'<span class="listing-category">{category}</span>'
        html += f'<span class="listing-status">{status}</span>'
        html += f'</div>'

        if description:
            html += f'<div class="listing-description">{description}</div>'

        # Seller info
        html += f'<div class="listing-seller">'
        html += f'<span class="seller-name">{seller_name}</span>'
        html += f'<span class="seller-rating">{"⭐" * int(seller_rating)}</span>'
        html += f'</div>'

        # Actions
        html += f'<div class="listing-actions">'
        if is_seller:
            html += f'<button class="btn-edit-listing">Edit Listing</button>'
            html += f'<button class="btn-mark-sold">Mark as Sold</button>'
        else:
            if status == "available":
                html += f'<button class="btn-buy-now">Buy Now</button>'
                html += f'<button class="btn-make-offer">Make Offer</button>'
                html += f'<button class="btn-contact-seller">Contact Seller</button>'
            else:
                html += f'<span class="listing-sold">This item is no longer available</span>'

        html += f'</div>'

        html += f'</div>'  # Close listing-details
        html += f'</div>'  # Close marketplace-listing

        return html


# Export plugin class
__all__ = ["MarketplacePlugin"]
